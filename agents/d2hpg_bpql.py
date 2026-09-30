import copy
import numpy as np
import torch
import torch.nn.functional as F
from replay_memory import ReplayMemory
from temporary_buffer import TemporaryBuffer
from core import GaussianPolicy, Critic
import utils as utils

class D2HPG_BPQL:
    def __init__(self, args, augmented_state_dim, state_dim, action_dim, action_bound, device):

        self.state_dim = state_dim
        self.augmented_state_dim = augmented_state_dim
        self.action_dim = action_dim
        self.device = device
        self.batch_size = args.batch_size
        self.action_bound = action_bound
        self.num_expl_steps = args.start_step

        self.update_type = args.update_type
        self.gamma = args.gamma
        self.critic_lr = args.critic_lr
        self.actor_lr = args.actor_lr
        self.alpha_lr = args.alpha_lr
        self.critic_target_tau = args.tau

        self.init_temperature = args.temperature
        self.lifting_wight = 0
        self.lift_w_max = 10.0  # 1.0 ~ 10.0
        self.lift_w_min = 0.0   # 0.0 ~ 1.0
        self.lift_hold  = int(args.max_step * 0.25)
        self.lift_decay = int(args.max_step * 0.75)
        self.lifting_repeat_obs = 100

        self.buffer = ReplayMemory(args.obs_delayed_steps, self.state_dim, self.action_dim, device, args.buffer_size)
        self.temporary_buffer = TemporaryBuffer(args.obs_delayed_steps)
        self.eval_temporary_buffer = TemporaryBuffer(args.obs_delayed_steps)
        self.abstract_state_dim  = self.state_dim
        self.abstract_action_dim = self.action_dim

        # Define actors
        self.actor = GaussianPolicy(args, self.augmented_state_dim, self.action_dim, self.action_bound).to(device)
        self.abstract_actor = GaussianPolicy(args, self.abstract_state_dim, self.abstract_action_dim, self.action_bound).to(device)
        self.abstract_actor_target = copy.deepcopy(self.abstract_actor)

        # Define BPQL critics
        self.critic = Critic(self.state_dim, self.action_dim, self.device).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.abstract_critic = Critic(self.abstract_state_dim, self.abstract_action_dim, self.device).to(device)
        self.abstract_critic_target = copy.deepcopy(self.abstract_critic).to(device)

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(self.action_dim)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.abstract_actor_optimizer = torch.optim.Adam(self.abstract_actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.abstract_critic_optimizer = torch.optim.Adam(self.abstract_critic.parameters(), lr=self.critic_lr)
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

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

    def train_critic(self, state, action, reward, discount, next_augmented_state, next_state, step):
        # Compute critic loss
        with torch.no_grad():
            next_actions, log_pi, _ = self.actor.sample(next_augmented_state)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_actions)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (discount * target_Q)

        # Get current Q estimates
        Q1, Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_abstract_critic(self, state, action, reward, discount, next_state, step):
        with torch.no_grad():
            abstract_policy_action, abstract_log_pi, _ = self.abstract_actor.sample(next_state)

            # Compute the target Q value
            target_Q1, target_Q2 = self.abstract_critic_target(next_state, abstract_policy_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha.detach() * abstract_log_pi
            target_Q = reward + (discount * target_Q)

        # Get current Q estimates
        Q1, Q2 = self.abstract_critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # Optimize the critic
        self.abstract_critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.abstract_critic_optimizer.step()

    def lifting_coef(self, step):
        if step <= self.lift_hold:
            return self.lift_w_max
        if step >= self.lift_decay:
            return self.lift_w_min
        t = (step - self.lift_hold) / float(self.lift_decay - self.lift_hold)
        return self.lift_w_max + (self.lift_w_min - self.lift_w_max) * t

    def train_actor(self, augmented_state, state, step):
        actions, log_pis, _ = self.actor.sample(augmented_state)
        q_values_A, q_values_B = self.critic(state, actions)
        q_values = torch.min(q_values_A, q_values_B)

        sac_loss = (self.log_alpha.exp().detach() * log_pis - q_values).mean()
        lift_loss = self.update_actor_by_lifting(augmented_state, state)

        if self.update_type == 'normal':
            actor_loss = sac_loss + self.lift_w_max * lift_loss
        elif self.update_type == 'decay':
            self.lifting_wight = self.lifting_coef(step)
            actor_loss = sac_loss + self.lifting_wight * lift_loss
        else:
            raise TypeError("update type error")

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_opt.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_pis + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_opt.step()

    def update_abstract_actor(self, states):
        abstract_pi, abstract_log_pi, _ = self.abstract_actor.sample(states)
        Q1, Q2 = self.abstract_critic(states, abstract_pi)
        Q = torch.min(Q1, Q2)

        abstract_actor_loss = (self.alpha.detach() * abstract_log_pi - Q).mean()

        # Optimize the abstract actor
        self.abstract_actor_optimizer.zero_grad(set_to_none=True)
        abstract_actor_loss.backward()
        self.abstract_actor_optimizer.step()

    def update_actor_by_lifting(self, augmented_states, states):
        batch_size = augmented_states.shape[0]

        obs_repeated = augmented_states.unsqueeze(1).repeat(1, self.lifting_repeat_obs, 1)
        obs_repeated = obs_repeated.view(int(batch_size * self.lifting_repeat_obs), -1)

        states_repeated = states.unsqueeze(1).repeat(1, self.lifting_repeat_obs, 1)
        states_repeated = states_repeated.view(batch_size * self.lifting_repeat_obs, -1)

        pi, _, _ = self.actor.sample(obs_repeated)
        with torch.no_grad():
            abstract_pi, _, _ = self.abstract_actor.sample(states_repeated)

        pi = pi.view(batch_size, self.lifting_repeat_obs, -1)
        abstract_pi = abstract_pi.view(batch_size, self.lifting_repeat_obs, -1)
        policy_lifting_loss = F.mse_loss(pi.mean(dim=1), abstract_pi.mean(dim=1))
        policy_lifting_loss += F.mse_loss(pi.std(dim=1), abstract_pi.std(dim=1))

        return policy_lifting_loss

    def train(self, step, i):
        augmented_states, actions, rewards, next_augmented_states, dones, states, next_states = self.buffer.sample(self.batch_size)
        discounts = (1 - dones) * self.gamma

        # update critics, actors
        self.update_abstract_critic(states, actions, rewards, discounts, next_states, step)
        self.update_abstract_actor(states)
        self.train_critic(states, actions, rewards, discounts, next_augmented_states, next_states, step)
        self.train_actor(augmented_states, states, step)

        # update target networks
        utils.soft_update(self.critic, self.critic_target, self.critic_target_tau)
        utils.soft_update(self.abstract_critic, self.abstract_critic_target, self.critic_target_tau)