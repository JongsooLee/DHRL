import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from utils import weight_init


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        args,
        state_dim,
        action_dim,
        action_bound,
        hidden_dims=(256, 256),
        activation_fc=F.relu,
        device="cuda",
    ):
        super().__init__()
        self.device = device  # Kept for backward compatibility.
        self.log_std_min = args.log_std_bound[0]
        self.log_std_max = args.log_std_bound[1]
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            for i in range(len(hidden_dims) - 1)
        )
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)

        low, high = _format_action_bounds(action_bound, action_dim)
        self.register_buffer("action_rescale", (high - low) / 2.0)
        self.register_buffer("action_rescale_bias", (high + low) / 2.0)
        if torch.any(self.action_rescale <= 0):
            raise ValueError("Every action upper bound must exceed its lower bound.")

        self.apply(weight_init)

    def _format(self, state):

        module_device = self.action_rescale.device
        if isinstance(state, torch.Tensor):
            x = state.to(device=module_device, dtype=torch.float32)
        else:
            x = torch.as_tensor(state, device=module_device, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        mean = self.mean_layer(x)
        log_std = torch.clamp(
            self.log_std_layer(x), self.log_std_min, self.log_std_max
        )
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        distribution = Normal(mean, log_std.exp())

        unbounded_action = distribution.rsample()
        bounded_action = torch.tanh(unbounded_action)
        action = bounded_action * self.action_rescale + self.action_rescale_bias

        jacobian = self.action_rescale * (1.0 - bounded_action.pow(2))
        log_prob = distribution.log_prob(unbounded_action) - torch.log(jacobian.clamp_min(1e-6))
        log_prob = log_prob.sum(dim=1, keepdim=True)
        deterministic_action = (torch.tanh(mean) * self.action_rescale + self.action_rescale_bias)
        return action, log_prob, deterministic_action


class Critic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        hidden_dims=(256, 256),
        activation_fc=F.relu,
    ):
        super().__init__()
        self.device = device
        self.activation_fc = activation_fc

        self.input_layer_A = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.hidden_layers_A = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            for i in range(len(hidden_dims) - 1)
        )
        self.output_layer_A = nn.Linear(hidden_dims[-1], 1)

        self.input_layer_B = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.hidden_layers_B = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            for i in range(len(hidden_dims) - 1)
        )
        self.output_layer_B = nn.Linear(hidden_dims[-1], 1)
        self.apply(weight_init)

    def _format(self, state, action):
        module_device = self.input_layer_A.weight.device
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, device=module_device, dtype=torch.float32)
        else:
            state = state.to(module_device, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, device=module_device, dtype=torch.float32)
        else:
            action = action.to(module_device, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        return state, action

    def forward(self, state, action):
        x, u = self._format(state, action)
        xu = torch.cat([x, u], dim=1)

        q1 = self.activation_fc(self.input_layer_A(xu))
        for hidden_layer in self.hidden_layers_A:
            q1 = self.activation_fc(hidden_layer(q1))
        q1 = self.output_layer_A(q1)

        q2 = self.activation_fc(self.input_layer_B(xu))
        for hidden_layer in self.hidden_layers_B:
            q2 = self.activation_fc(hidden_layer(q2))
        q2 = self.output_layer_B(q2)
        return q1, q2


# ============================================================================
# Principled D2HPG components
# ============================================================================

class StateEncoder(nn.Module):
    def __init__(self, augmented_state_dim, abstract_state_dim, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(augmented_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, abstract_state_dim),
        )
        self.apply(weight_init)

    def forward(self, augmented_state):
        return self.encoder(augmented_state)


class ActionEncoder(nn.Module):

    def __init__(
        self,
        augmented_state_dim,
        action_dim,
        abstract_action_dim,
        action_bound,
        hidden_dim=256,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(augmented_state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, abstract_action_dim),
        )

        if abstract_action_dim == action_dim:
            low, high = _format_action_bounds(action_bound, action_dim)
        else:
            low = -torch.ones(abstract_action_dim, dtype=torch.float32)
            high = torch.ones(abstract_action_dim, dtype=torch.float32)
        self.register_buffer("action_rescale", (high - low) / 2.0)
        self.register_buffer("action_rescale_bias", (high + low) / 2.0)
        self.apply(weight_init)

    def forward(self, augmented_state, action):
        state_action = torch.cat([augmented_state, action], dim=1)
        normalized_abstract_action = torch.tanh(self.encoder(state_action))
        return normalized_abstract_action * self.action_rescale + self.action_rescale_bias


class AbstractRewardModel(nn.Module):
    def __init__(self, abstract_state_dim, abstract_action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(abstract_state_dim + abstract_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(weight_init)

    def forward(self, abstract_state, abstract_action):
        return self.network(torch.cat([abstract_state, abstract_action], dim=1))


class AbstractTransitionModel(nn.Module):
    def __init__(
        self,
        abstract_state_dim,
        abstract_action_dim,
        hidden_dim=256,
        log_std_min=-5.0,
        log_std_max=2.0,
    ):
        super().__init__()
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        if self.log_std_max <= self.log_std_min:
            raise ValueError("transition log-std upper bound must exceed lower bound")

        self.trunk = nn.Sequential(
            nn.Linear(abstract_state_dim + abstract_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, abstract_state_dim)
        self.log_std_head = nn.Linear(hidden_dim, abstract_state_dim)
        self.apply(weight_init)

    def forward(self, abstract_state, abstract_action):
        hidden = self.trunk(torch.cat([abstract_state, abstract_action], dim=1))
        mean = self.mean_head(hidden)
        log_std = torch.clamp(self.log_std_head(hidden), self.log_std_min, self.log_std_max)
        return mean, log_std.exp()

    def rsample(self, abstract_state, abstract_action, generator=None):
        mean, std = self.forward(abstract_state, abstract_action)
        noise = torch.randn(std.shape, device=std.device, dtype=std.dtype, generator=generator)
        return mean + std * noise, mean, std


def diagonal_gaussian_w2(mean_1, std_1, mean_2, std_2, eps=1e-8):

    squared_distance = (mean_1 - mean_2).pow(2).sum(dim=1) + (std_1 - std_2).pow(2).sum(dim=1)
    return torch.sqrt(squared_distance + eps)

def _format_action_bounds(action_bound, action_dim):

    if len(action_bound) != 2:
        raise ValueError("action_bound must be [low, high]")
    low = torch.as_tensor(np.asarray(action_bound[0]), dtype=torch.float32)
    high = torch.as_tensor(np.asarray(action_bound[1]), dtype=torch.float32)
    if low.ndim == 0:
        low = low.repeat(action_dim)
    if high.ndim == 0:
        high = high.repeat(action_dim)
    low = low.reshape(-1)
    high = high.reshape(-1)
    if low.numel() != action_dim or high.numel() != action_dim:
        raise ValueError(
            f"Action bounds must each have {action_dim} entries; "
            f"got {low.numel()} and {high.numel()}."
        )
    return low, high
