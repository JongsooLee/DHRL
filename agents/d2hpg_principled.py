import copy
import numpy as np
import torch
import torch.nn.functional as F

from core import (
    AbstractRewardModel,
    AbstractTransitionModel,
    Critic,
    GaussianPolicy,
    StateEncoder,
    diagonal_gaussian_w2,
)
from replay_memory import ReplayMemory
from temporary_buffer import TemporaryBuffer
import utils


class D2HPG_Principled:

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
        self.lifting_repeat_obs = args.lifting_repeat_obs
        self.lifting_std_unbiased = (
            bool(args.lifting_std_unbiased)
            if self.comparison_mode == "controlled"
            else False
        )

        self.lift_w_max = args.lift_w_max
        self.lift_w_min = args.lift_w_min
        self.lift_hold = int(args.max_step * args.lift_hold_ratio)
        self.lift_decay = int(args.max_step * args.lift_decay_ratio)

        self.principled_lift_weight = float(args.principled_lift_weight)
        self.principled_lifting_start_step = int(args.principled_lifting_start_step)
        self.principled_lifting_ramp_steps = max(1, int(args.principled_lifting_ramp_steps))

        self.abstract_state_dim = (state_dim if args.abstract_state_dim <= 0 else args.abstract_state_dim)
        self.abstract_action_dim = (action_dim if args.abstract_action_dim <= 0 else args.abstract_action_dim)
        if self.abstract_action_dim != self.action_dim:
            raise ValueError("Identity action mapping requires abstract_action_dim == action_dim. Use --abstract-action-dim 0.")

        self.buffer = ReplayMemory(
            args.obs_delayed_steps,
            self.state_dim,
            self.action_dim,
            device,
            args.buffer_size,
        )
        self.temporary_buffer = TemporaryBuffer(args.obs_delayed_steps)
        self.eval_temporary_buffer = TemporaryBuffer(args.obs_delayed_steps)

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

        self.use_abstract_actor_target = (
            bool(args.controlled_use_abstract_actor_target)
            if self.comparison_mode == "controlled"
            else True
        )
        self.abstract_actor_target = (
            copy.deepcopy(self.abstract_actor).to(device)
            if self.use_abstract_actor_target
            else None
        )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(args.random_seed) + 104729)
            self.state_encoder = StateEncoder(
                self.augmented_state_dim,
                self.abstract_state_dim,
                hidden_dim=args.homomorphism_hidden_dim,
            ).to(device)
            self.abstract_reward_model = AbstractRewardModel(
                self.abstract_state_dim,
                self.abstract_action_dim,
                hidden_dim=args.homomorphism_hidden_dim,
            ).to(device)
            self.abstract_transition_model = AbstractTransitionModel(
                self.abstract_state_dim,
                self.abstract_action_dim,
                hidden_dim=args.homomorphism_hidden_dim,
                log_std_min=args.transition_log_std_bound[0],
                log_std_max=args.transition_log_std_bound[1],
            ).to(device)

        self.lax_bisim_coef = float(args.lax_bisim_coef)
        self.transition_coef = float(args.transition_coef)
        self.reward_model_coef = float(args.reward_model_coef)
        self.transition_loss_type = str(args.transition_loss_type)
        self.detach_next_abstract_state = bool(args.detach_next_abstract_state)
        self.detach_lax_target = bool(args.detach_lax_target)
        self.abstract_critic_reward_source = str(args.abstract_critic_reward_source)
        self.homomorphism_gradient_clip = float(args.homomorphism_gradient_clip)
        self.show_principled_loss = bool(args.show_principled_loss)

        if self.transition_loss_type not in {"gaussian_nll", "sample_mse"}:
            raise ValueError("Unknown transition loss type")
        if self.abstract_critic_reward_source not in {"environment", "model"}:
            raise ValueError("Unknown abstract critic reward source")

        self.log_alpha = torch.tensor(
            np.log(self.init_temperature),
            device=device,
            dtype=torch.float32,
            requires_grad=self.automatic_entropy_tuning,
        )
        self.target_entropy = -float(self.action_dim)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.abstract_actor_optimizer = torch.optim.Adam(self.abstract_actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.abstract_critic_optimizer = torch.optim.Adam(self.abstract_critic.parameters(), lr=self.critic_lr)
        self.log_alpha_opt = (torch.optim.Adam([self.log_alpha], lr=self.alpha_lr) if self.automatic_entropy_tuning else None)

        self.homomorphism_parameters = (
            list(self.state_encoder.parameters())
            + list(self.abstract_reward_model.parameters())
            + list(self.abstract_transition_model.parameters())
        )
        self.homomorphism_optimizer = torch.optim.Adam(self.homomorphism_parameters, lr=args.homomorphism_lr)

        generator_device = "cuda" if device.type == "cuda" else "cpu"
        self.homomorphism_generator = torch.Generator(device=generator_device)
        self.homomorphism_generator.manual_seed(int(args.random_seed) + 130363)
        self.last_principled_losses = {}

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
        return (
            self.abstract_actor_target
            if self.use_abstract_actor_target
            else self.abstract_actor
        )

    def _derangement(self, batch_size, device):
        if batch_size < 2:
            raise ValueError("L_lax requires batch_size >= 2")
        shift = int(torch.randint(1, batch_size, (1,), generator=self.homomorphism_generator, device=device).item())
        return torch.roll(torch.arange(batch_size, device=device), shifts=shift)

    # ======================================================================
    # Learn h=(f, identity), R_bar, and P_bar using L_lax + L_h.
    # ======================================================================
    def update_homomorphism(self, augmented_state, action, reward, next_augmented_state):
        abstract_state = self.state_encoder(augmented_state)
        next_abstract_state = self.state_encoder(next_augmented_state)
        next_target = (next_abstract_state.detach() if self.detach_next_abstract_state else next_abstract_state)

        abstract_action = action
        transition_mean, transition_std = self.abstract_transition_model(abstract_state, abstract_action)
        predicted_reward = self.abstract_reward_model(abstract_state, abstract_action)

        if self.transition_loss_type == "gaussian_nll":
            safe_std = transition_std.clamp_min(1e-6)
            normalized_error = (transition_mean - next_target) / safe_std
            transition_loss = (0.5 * normalized_error.pow(2) + torch.log(safe_std)).mean()
        else:
            noise = torch.randn(transition_std.shape, device=transition_std.device, dtype=transition_std.dtype, generator=self.homomorphism_generator)
            transition_sample = transition_mean + transition_std * noise
            transition_loss = (transition_sample - next_target).pow(2).sum(dim=1).mean()

        reward_model_loss = F.mse_loss(predicted_reward, reward)

        permutation = self._derangement(abstract_state.shape[0], abstract_state.device)
        state_distance = (abstract_state - abstract_state[permutation]).abs().sum(dim=1)
        reward_distance = (reward.squeeze(1) - reward[permutation].squeeze(1)).abs()
        transition_distance = diagonal_gaussian_w2(transition_mean, transition_std, transition_mean[permutation], transition_std[permutation])
        lax_target = reward_distance + self.gamma * transition_distance
        if self.detach_lax_target:
            lax_target = lax_target.detach()
        lax_loss = (state_distance - lax_target).pow(2).mean()

        homomorphism_loss = (self.lax_bisim_coef * lax_loss + self.transition_coef * transition_loss + self.reward_model_coef * reward_model_loss)

        self.homomorphism_optimizer.zero_grad(set_to_none=True)
        homomorphism_loss.backward()
        if self.homomorphism_gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.homomorphism_parameters, self.homomorphism_gradient_clip)
        self.homomorphism_optimizer.step()

        losses = {
            "homomorphism": homomorphism_loss.detach(),
            "lax": lax_loss.detach(),
            "transition": transition_loss.detach(),
            "reward_model": reward_model_loss.detach(),
            "abstract_state_std": abstract_state.std(
                dim=0, unbiased=False
            ).mean().detach(),
            "abstract_action_std": abstract_action.std(
                dim=0, unbiased=False
            ).mean().detach(),
            "transition_std": transition_std.mean().detach(),
        }
        self.last_principled_losses = losses
        return losses

    # ======================================================================
    # Regular SAC: same operations as D2HPG_SAC.
    # ======================================================================
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
            target_q1, target_q2 = self.critic_target(next_augmented_state, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * log_pi
            target_q = reward + discount * target_q

        q1, q2 = self.critic(augmented_state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.detach()

    # ======================================================================
    # Abstract actor-critic on projected replay transitions.
    # ======================================================================
    def make_abstract_transition(self, augmented_state, action, reward, next_augmented_state):
        with torch.no_grad():
            abstract_state = self.state_encoder(augmented_state)
            abstract_action = action
            next_abstract_state = self.state_encoder(next_augmented_state)
            if self.abstract_critic_reward_source == "model":
                abstract_reward = self.abstract_reward_model(abstract_state, abstract_action)
            else:
                abstract_reward = reward
        return (abstract_state, abstract_action, abstract_reward, next_abstract_state)

    def update_abstract_critic(
        self,
        abstract_state,
        abstract_action,
        abstract_reward,
        discount,
        next_abstract_state,
        step,
    ):
        with torch.no_grad():
            next_action, log_pi, _ = self._abstract_teacher().sample(next_abstract_state)
            target_q1, target_q2 = self.abstract_critic_target(next_abstract_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * log_pi
            target_q = abstract_reward + discount * target_q

        q1, q2 = self.abstract_critic(abstract_state, abstract_action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.abstract_critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.abstract_critic_optimizer.step()
        return critic_loss.detach()

    def update_abstract_actor(self, abstract_states):
        abstract_pi, abstract_log_pi, _ = self.abstract_actor.sample(abstract_states)
        q1, q2 = self.abstract_critic(abstract_states, abstract_pi)
        actor_loss = (self.alpha.detach() * abstract_log_pi - torch.min(q1, q2)).mean()
        self.abstract_actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.abstract_actor_optimizer.step()
        return actor_loss.detach()

    # ======================================================================
    # Policy lifting.
    # ======================================================================
    def lifting_coef(self, step):
        if self.comparison_mode == "controlled":
            if step <= self.lift_hold:
                return self.lift_w_max
            if step >= self.lift_decay:
                return self.lift_w_min
            progress = (step - self.lift_hold) / float(self.lift_decay - self.lift_hold)
            return self.lift_w_max + (self.lift_w_min - self.lift_w_max) * progress

        if step < self.principled_lifting_start_step:
            return 0.0
        if self.update_type == "normal":
            return self.principled_lift_weight
        progress = (step - self.principled_lifting_start_step) / float(self.principled_lifting_ramp_steps)
        return self.principled_lift_weight * min(1.0, max(0.0, progress))

    def update_actor_by_lifting(self, augmented_states):
        batch_size = augmented_states.shape[0]
        repeated_states = augmented_states.unsqueeze(1).repeat(1, self.lifting_repeat_obs, 1)
        repeated_states = repeated_states.view(batch_size * self.lifting_repeat_obs, -1)

        regular_actions, _, _ = self.actor.sample(repeated_states)
        transformed_actions = regular_actions

        with torch.no_grad():
            abstract_states = self.state_encoder(repeated_states)
            abstract_actions, _, _ = self._abstract_teacher().sample(abstract_states)

        transformed_actions = transformed_actions.view(batch_size, self.lifting_repeat_obs, -1)
        abstract_actions = abstract_actions.view(batch_size, self.lifting_repeat_obs, -1)
        lifting_loss = F.mse_loss(transformed_actions.mean(dim=1), abstract_actions.mean(dim=1))
        lifting_loss += F.mse_loss(transformed_actions.std(dim=1, unbiased=self.lifting_std_unbiased), abstract_actions.std(dim=1, unbiased=self.lifting_std_unbiased))

        return lifting_loss

    def train_actor(self, augmented_state, step):
        actions, log_pis, _ = self.actor.sample(augmented_state)
        q1, q2 = self.critic(augmented_state, actions)
        sac_loss = (self.alpha.detach() * log_pis - torch.min(q1, q2)).mean()
        lifting_loss = self.update_actor_by_lifting(augmented_state)

        if self.comparison_mode == "controlled":
            if self.update_type == "normal":
                self.lifting_wight = self.lift_w_max
            elif self.update_type == "decay":
                self.lifting_wight = self.lifting_coef(step)
            else:
                raise TypeError("update type error")
        else:
            self.lifting_wight = self.lifting_coef(step)

        actor_loss = sac_loss + self.lifting_wight * lifting_loss
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.zeros((), device=self.device)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha.exp() * (log_pis + self.target_entropy).detach()).mean()
            self.log_alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.log_alpha_opt.step()

        return (actor_loss.detach(), sac_loss.detach(), lifting_loss.detach(), torch.tensor(self.lifting_wight, device=self.device))

    def train(self, step, i):
        augmented_states, actions, rewards, next_augmented_states, dones, states, next_states  = self.buffer.sample(self.batch_size)
        discounts = (1.0 - dones) * self.gamma

        del states, next_states

        principled_losses = self.update_homomorphism(augmented_states, actions, rewards, next_augmented_states)
        abstract_states, abstract_actions, abstract_rewards, next_abstract_states = self.make_abstract_transition(augmented_states, actions, rewards, next_augmented_states)

        abstract_critic_loss = self.update_abstract_critic(abstract_states, abstract_actions, abstract_rewards, discounts, next_abstract_states, step)
        abstract_actor_loss = self.update_abstract_actor(abstract_states)
        regular_critic_loss = self.train_critic(augmented_states, actions, rewards, discounts, next_augmented_states, step)
        actor_loss, sac_loss, lifting_loss, lifting_coefficient = self.train_actor(augmented_states, step)

        utils.soft_update(self.critic, self.critic_target, self.critic_target_tau)
        utils.soft_update(self.abstract_critic, self.abstract_critic_target, self.critic_target_tau)
        if self.use_abstract_actor_target:
            utils.soft_update(self.abstract_actor, self.abstract_actor_target, self.critic_target_tau)

        if self.show_principled_loss and i == 0 and step % 5000 == 0:
            print(
                "Principled losses | step {} | "
                "L_homo {:.4f} | L_lax {:.4f} | L_trans {:.4f} | "
                "L_reward {:.4f} | z_std {:.4f} | action_std {:.4f} | "
                "P_std {:.4f} | Q_abs {:.4f} | Pi_abs {:.4f} | "
                "Q_reg {:.4f} | Pi_reg {:.4f} | SAC {:.4f} | "
                "L_lift {:.4f} | lambda_lift {:.4f}".format(
                    step,
                    principled_losses["homomorphism"].item(),
                    principled_losses["lax"].item(),
                    principled_losses["transition"].item(),
                    principled_losses["reward_model"].item(),
                    principled_losses["abstract_state_std"].item(),
                    principled_losses["abstract_action_std"].item(),
                    principled_losses["transition_std"].item(),
                    abstract_critic_loss.item(),
                    abstract_actor_loss.item(),
                    regular_critic_loss.item(),
                    actor_loss.item(),
                    sac_loss.item(),
                    lifting_loss.item(),
                    lifting_coefficient.item(),
                )
            )
