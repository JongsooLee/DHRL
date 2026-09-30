import copy
from temporary_buffer import TemporaryBuffer
import numpy as np
import torch
import torch.nn.functional as F
from replay_memory import ReplayMemory
from core import GaussianPolicy, Critic
import utils as utils

class D2HPG_Naive:
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

        self.init_temperature   = args.temperature
        self.lifting_repeat_obs = 100
        self.lifting_weight = 10.0 # 1.0 ~ 10.0
        self.buffer = ReplayMemory(args.obs_delayed_steps, self.state_dim, self.action_dim, device, args.buffer_size)
        self.temporary_buffer = TemporaryBuffer(args.obs_delayed_steps)
        self.eval_temporary_buffer = TemporaryBuffer(args.obs_delayed_steps)
        self.abstract_state_dim  = self.state_dim
        self.abstract_action_dim = self.action_dim

        # Define actors
        self.actor = GaussianPolicy(args, self.augmented_state_dim, self.action_dim, self.action_bound).to(device)
        self.abstract_actor = GaussianPolicy(args, self.abstract_state_dim, self.abstract_action_dim, self.action_bound).to(device)
        self.abstract_actor_target = copy.deepcopy(self.abstract_actor)

        # Define critics
        self.abstract_critic = Critic(self.abstract_state_dim, self.abstract_action_dim, self.device).to(device)
        self.abstract_critic_target = copy.deepcopy(self.abstract_critic).to(device)

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(self.action_dim)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.abstract_actor_optimizer = torch.optim.Adam(self.abstract_actor.parameters(), lr=self.actor_lr)
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
        policy_lifting_loss = self.lifting_weight * policy_lifting_loss

        # Optimize the actual actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_lifting_loss.backward()
        self.actor_optimizer.step()

    def train(self, step, i):
        augmented_states, actions, rewards, next_augmented_states, dones, states, next_states = self.buffer.sample(self.batch_size)
        discounts = (1 - dones) * self.gamma

        # update critics, actors
        self.update_abstract_critic(states, actions, rewards, discounts, next_states, step)
        self.update_abstract_actor(states)
        self.update_actor_by_lifting(augmented_states, states)

        # update target networks
        utils.soft_update(self.abstract_critic, self.abstract_critic_target, self.critic_target_tau)
