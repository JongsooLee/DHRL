import copy

import numpy as np
import torch
import torch.nn.functional as F

from core import Critic, GaussianPolicy
from replay_memory import ReplayMemory
from temporary_buffer import TemporaryBuffer
import utils


class D2HPG_SAC:
    def __init__(
        self,
        args,
        augmented_state_dim,
        state_dim,
        action_dim,
        action_bound,
        device,
    ):
        self.state_dim = state_dim
        self.augmented_state_dim = augmented_state_dim
        self.action_dim = action_dim
        self.device = device
        self.batch_size = args.batch_size
        self.action_bound = action_bound
        self.num_expl_steps = args.start_step

        self.comparison_mode = getattr(args, "comparison_mode", "submitted")
        self.update_type = args.update_type
        self.gamma = args.gamma
        self.critic_lr = args.critic_lr
        self.actor_lr = args.actor_lr
        self.alpha_lr = args.alpha_lr
        self.critic_target_tau = args.tau

        self.init_temperature = args.temperature
        self.automatic_entropy_tuning = bool(args.automating_temperature)
        self.lifting_wight = 0.0
        self.lift_w_max = args.lift_w_max
        self.lift_w_min = args.lift_w_min
        self.lift_hold = int(args.max_step * args.lift_hold_ratio)
        self.lift_decay = int(args.max_step * args.lift_decay_ratio)
        self.lifting_repeat_obs = args.lifting_repeat_obs

        self.use_abstract_actor_target = (
            self.comparison_mode == "controlled"
            and bool(args.controlled_use_abstract_actor_target)
        )
        self.lifting_std_unbiased = (
            bool(args.lifting_std_unbiased)
            if self.comparison_mode == "controlled"
            else True
        )

        self.buffer = ReplayMemory(args.obs_delayed_steps, self.state_dim, self.action_dim, device, args.buffer_size)
        self.temporary_buffer = TemporaryBuffer(args.obs_delayed_steps)
        self.eval_temporary_buffer = TemporaryBuffer(args.obs_delayed_steps)
        self.abstract_state_dim = self.state_dim
        self.abstract_action_dim = self.action_dim

        self.actor = GaussianPolicy(
            args,
            self.augmented_state_dim,
            self.action_dim,
            self.action_bound,
            device=device,
        ).to(device)
        self.abstract_actor = GaussianPolicy(
            args,
            self.abstract_state_dim,
            self.abstract_action_dim,
            self.action_bound,
            device=device,
        ).to(device)

        self.critic = Critic(self.augmented_state_dim, self.action_dim, self.device).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.abstract_critic = Critic(self.abstract_state_dim, self.abstract_action_dim, self.device).to(device)
        self.abstract_critic_target = copy.deepcopy(self.abstract_critic).to(device)

        self.abstract_actor_target = (copy.deepcopy(self.abstract_actor).to(device) if self.use_abstract_actor_target else None)

        self.log_alpha = torch.tensor(np.log(self.init_temperature), device=device, dtype=torch.float32, requires_grad=self.automatic_entropy_tuning)
        self.target_entropy = -float(self.action_dim)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.abstract_actor_optimizer = torch.optim.Adam(self.abstract_actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.abstract_critic_optimizer = torch.optim.Adam(self.abstract_critic.parameters(), lr=self.critic_lr)
        self.log_alpha_opt = (torch.optim.Adam([self.log_alpha], lr=self.alpha_lr) if self.automatic_entropy_tuning else None)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(self, augmented_state, evaluation=True):
        with torch.no_grad():
            if evaluation:
                _, _, action = self.actor.sample(augmented_state)
            else:
                action, _, _ = self.actor.sample(augmented_state)
        return action.cpu().numpy()[0]

    def _abstract_teacher(self):
        return self.abstract_actor_target if self.use_abstract_actor_target else self.abstract_actor

    def train_critic(
        self,
        augmented_state,
        action,
        reward,
        discount,
        next_augmented_state,
        step,
    ):
        with torch.no_grad():
            next_actions, log_pi, _ = self.actor.sample(next_augmented_state)
            target_q1, target_q2 = self.critic_target(
                next_augmented_state, next_actions
            )
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * log_pi
            target_q = reward + discount * target_q

        q1, q2 = self.critic(augmented_state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.detach()

    def update_abstract_critic(
        self, state, action, reward, discount, next_state, step
    ):
        with torch.no_grad():
            abstract_policy_action, abstract_log_pi, _ = self._abstract_teacher().sample(next_state)
            target_q1, target_q2 = self.abstract_critic_target(next_state, abstract_policy_action)
            target_q = (torch.min(target_q1, target_q2) - self.alpha.detach() * abstract_log_pi)
            target_q = reward + discount * target_q

        q1, q2 = self.abstract_critic(state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.abstract_critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.abstract_critic_optimizer.step()
        return critic_loss.detach()

    def lifting_coef(self, step):
        if step <= self.lift_hold:
            return self.lift_w_max
        if step >= self.lift_decay:
            return self.lift_w_min
        progress = (step - self.lift_hold) / float(self.lift_decay - self.lift_hold)
        return self.lift_w_max + (self.lift_w_min - self.lift_w_max) * progress

    def train_actor(self, augmented_state, state, step):
        actions, log_pis, _ = self.actor.sample(augmented_state)
        q1, q2 = self.critic(augmented_state, actions)
        sac_loss = (self.alpha.detach() * log_pis - torch.min(q1, q2)).mean()
        lift_loss = self.update_actor_by_lifting(augmented_state, state)

        if self.update_type == "normal":
            self.lifting_wight = self.lift_w_max
        elif self.update_type == "decay":
            self.lifting_wight = self.lifting_coef(step)
        else:
            raise TypeError("update type error")
        actor_loss = sac_loss + self.lifting_wight * lift_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.zeros((), device=self.device)
        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha.exp()
                * (log_pis + self.target_entropy).detach()
            ).mean()
            self.log_alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.log_alpha_opt.step()
        return actor_loss.detach(), sac_loss.detach(), lift_loss.detach()

    def update_abstract_actor(self, states):
        abstract_pi, abstract_log_pi, _ = self.abstract_actor.sample(states)
        q1, q2 = self.abstract_critic(states, abstract_pi)
        abstract_actor_loss = (
            self.alpha.detach() * abstract_log_pi - torch.min(q1, q2)
        ).mean()

        self.abstract_actor_optimizer.zero_grad(set_to_none=True)
        abstract_actor_loss.backward()
        self.abstract_actor_optimizer.step()
        return abstract_actor_loss.detach()

    def update_actor_by_lifting(self, augmented_states, states):
        batch_size = augmented_states.shape[0]

        obs_repeated = augmented_states.unsqueeze(1).repeat(1, self.lifting_repeat_obs, 1)
        obs_repeated = obs_repeated.view(batch_size * self.lifting_repeat_obs, -1)

        states_repeated = states.unsqueeze(1).repeat(1, self.lifting_repeat_obs, 1)
        states_repeated = states_repeated.view(batch_size * self.lifting_repeat_obs, -1)

        pi, _, _ = self.actor.sample(obs_repeated)
        with torch.no_grad():
            abstract_pi, _, _ = self._abstract_teacher().sample(states_repeated)

        pi = pi.view(batch_size, self.lifting_repeat_obs, -1)
        abstract_pi = abstract_pi.view(batch_size, self.lifting_repeat_obs, -1)
        policy_lifting_loss = F.mse_loss(pi.mean(dim=1), abstract_pi.mean(dim=1))
        policy_lifting_loss += F.mse_loss(pi.std(dim=1, unbiased=self.lifting_std_unbiased),
                                          abstract_pi.std(dim=1, unbiased=self.lifting_std_unbiased))
        return policy_lifting_loss

    def train(self, step, i):
        (
            augmented_states,
            actions,
            rewards,
            next_augmented_states,
            dones,
            states,
            next_states,
        ) = self.buffer.sample(self.batch_size)
        discounts = (1.0 - dones) * self.gamma

        self.update_abstract_critic(states, actions, rewards, discounts, next_states, step)
        self.update_abstract_actor(states)
        self.train_critic(
            augmented_states,
            actions,
            rewards,
            discounts,
            next_augmented_states,
            step,
        )
        self.train_actor(augmented_states, states, step)

        utils.soft_update(self.critic, self.critic_target, self.critic_target_tau)
        utils.soft_update(self.abstract_critic, self.abstract_critic_target, self.critic_target_tau)
        if self.use_abstract_actor_target:
            utils.soft_update(self.abstract_actor, self.abstract_actor_target, self.critic_target_tau)
