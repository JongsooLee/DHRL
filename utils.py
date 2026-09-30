import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from wrapper import DelayedEnv


def weight_init(m):
    """Orthogonal initialization used by the submitted implementation."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def hard_update(network, target_network):
    with torch.no_grad():
        for param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data.copy_(param.data)


def soft_update(network, target_network, tau):
    with torch.no_grad():
        for param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )


def set_seed(random_seed):
    if random_seed <= 0:
        random_seed = int(np.random.randint(1, 9999))

    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return random_seed


def make_env(env_name, random_seed):
    import gym

    env = gym.make(env_name)
    env.reset(seed=random_seed)
    env.action_space.seed(random_seed)

    eval_env = gym.make(env_name)
    eval_env.reset(seed=random_seed + 1000)
    eval_env.action_space.seed(random_seed + 1000)
    return env, eval_env


def make_delayed_env(args, random_seed, obs_delayed_steps, act_delayed_steps):
    import gym

    env = gym.make(args.env_name)
    eval_env = gym.make(args.env_name)

    delayed_env = DelayedEnv(
        env,
        seed=random_seed,
        obs_delayed_steps=obs_delayed_steps,
        act_delayed_steps=act_delayed_steps,
    )
    eval_delayed_env = DelayedEnv(
        eval_env,
        seed=random_seed + 1000,
        obs_delayed_steps=obs_delayed_steps,
        act_delayed_steps=act_delayed_steps,
    )
    return delayed_env, eval_delayed_env


def get_log_path(args, random_seed):
    mode = getattr(args, "comparison_mode", "submitted")
    delay = int(args.obs_delayed_steps)
    filename = (
        f"{mode}_{args.agent_type}_{args.env_name}_"
        f"delay{delay}_seed{int(random_seed)}.txt"
    )
    return Path(getattr(args, "log_dir", "./log")) / filename


def initialize_log_file(args, random_seed):
    """Prepare a clean reward log for one run."""
    path = get_log_path(args, random_seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if getattr(args, "overwrite_log", True):
        path.write_text("")
    return path


def log_to_txt(args, total_step, result, random_seed):
    path = get_log_path(args, random_seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(f"{int(total_step)} {float(result)}\n")


def write_run_summary(args, random_seed, summary):
    path = get_log_path(args, random_seed).with_suffix(".summary.json")
    serializable = dict(summary)
    serializable.update(
        {
            "agent_type": args.agent_type,
            "comparison_mode": getattr(args, "comparison_mode", "submitted"),
            "env_name": args.env_name,
            "obs_delayed_steps": int(args.obs_delayed_steps),
            "random_seed": int(random_seed),
        }
    )
    path.write_text(json.dumps(serializable, indent=2))
