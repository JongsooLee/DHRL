import argparse

import numpy as np
import torch

from agents.d2hpg_bpql import D2HPG_BPQL
from agents.d2hpg_naive import D2HPG_Naive
from agents.d2hpg_principled import D2HPG_Principled
from agents.d2hpg_sac import D2HPG_SAC
from trainer import Trainer
from utils import make_delayed_env, set_seed


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def get_parameters(delay):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="Ant-v3", type=str)
    parser.add_argument("--obs-delayed-steps", default=delay, type=int)
    parser.add_argument("--act-delayed-steps", default=0, type=int)
    parser.add_argument("--random-seed", default=1, type=int)
    parser.add_argument("--num-trials", default=10, type=int)
    parser.add_argument("--log-dir", default="./log", type=str)
    parser.add_argument("--overwrite-log", default=True, type=str2bool)
    parser.add_argument("--eval-flag", default=True, type=str2bool)
    parser.add_argument("--eval-freq", default=5000, type=int)
    parser.add_argument("--eval-episode", default=10, type=int)
    parser.add_argument("--automating-temperature", default=True, type=str2bool)
    parser.add_argument("--temperature", default=0.2, type=float)
    parser.add_argument("--start-step", default=10000, type=int)
    parser.add_argument("--max-step", default=1000000, type=int)
    parser.add_argument("--update-after", default=100, type=int)
    parser.add_argument("--hidden-dims", default=(256, 256), nargs="+", type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--buffer-size", default=1000000, type=int)
    parser.add_argument("--update-every", default=50, type=int)
    parser.add_argument("--log-std-bound", default=[-20.0, 2.0], type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--critic-lr", default=3e-4, type=float)
    parser.add_argument("--actor-lr", default=3e-4, type=float)
    parser.add_argument("--alpha-lr", default=3e-4, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--agent-type", default="sac", choices=["naive", "sac", "bpql", "principled"])
    parser.add_argument("--update-type", default="decay", choices=["normal", "decay"])

    # controlled matches practical/principled comparison settings.
    # Identity g(x,a)=a is used by principled D2HPG in both modes.
    parser.add_argument("--comparison-mode", default="controlled", choices=["submitted", "controlled"])

    parser.add_argument("--abstract-state-dim", default=0, type=int)
    parser.add_argument("--abstract-action-dim", default=0, type=int)
    parser.add_argument("--homomorphism-hidden-dim", default=256, type=int)
    parser.add_argument("--homomorphism-lr", default=3e-4, type=float)
    parser.add_argument("--lax-bisim-coef", default=1.0, type=float)
    parser.add_argument("--transition-coef", default=1.0, type=float)
    parser.add_argument("--reward-model-coef", default=1.0, type=float)
    parser.add_argument("--transition-log-std-bound", default=[-5.0, 2.0], type=float)
    parser.add_argument("--transition-loss-type", default="gaussian_nll", choices=["gaussian_nll", "sample_mse"])
    parser.add_argument("--detach-next-abstract-state", default=True, type=str2bool)
    parser.add_argument("--detach-lax-target", default=True, type=str2bool)
    parser.add_argument("--abstract-critic-reward-source", default="environment", choices=["environment", "model"])
    parser.add_argument("--homomorphism-gradient-clip", default=10.0, type=float)
    parser.add_argument("--show-principled-loss", default=False, type=str2bool)

    # Common policy lifting configuration.
    parser.add_argument("--lift-w-max", default=10.0, type=float)
    parser.add_argument("--lift-w-min", default=0.0, type=float)
    parser.add_argument("--lift-hold-ratio", default=0.25, type=float)
    parser.add_argument("--lift-decay-ratio", default=0.75, type=float)
    parser.add_argument("--lifting-repeat-obs", default=100, type=int)
    parser.add_argument("--lifting-std-unbiased", default=True, type=str2bool)
    parser.add_argument(
        "--controlled-use-abstract-actor-target",
        default=False,
        type=str2bool,
        help=(
            "Use the same target abstract actor in both practical and "
            "principled agents. False minimally changes the submitted SAC code."
        ),
    )

    # Legacy principled-only schedule retained only for submitted mode.
    parser.add_argument("--principled-lift-weight", default=0.5, type=float)
    parser.add_argument("--principled-lifting-start-step", default=10000, type=int)
    parser.add_argument("--principled-lifting-ramp-steps", default=20000, type=int)

    return parser.parse_args()

def run_one(args, random_seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_seed = set_seed(random_seed)

    env, eval_env = make_delayed_env(
        args,
        random_seed,
        obs_delayed_steps=args.obs_delayed_steps,
        act_delayed_steps=args.act_delayed_steps,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    augmented_state_dim = state_dim + action_dim * args.obs_delayed_steps
    action_bound = [np.asarray(env.action_space.low, dtype=np.float32).copy(), np.asarray(env.action_space.high, dtype=np.float32).copy()]

    print(
        f"Environment: {args.env_name}, "
        f"Obs. Delayed Steps: {args.obs_delayed_steps}, "
        f"Random Seed: {random_seed}, "
        f"Agent: {args.agent_type}, "
        f"Mode: {args.comparison_mode}, "
        f"Device: {device}\n"
    )

    agent_classes = {
        "naive": D2HPG_Naive,
        "sac": D2HPG_SAC,
        "bpql": D2HPG_BPQL,
        "principled": D2HPG_Principled,
    }
    agent = agent_classes[args.agent_type](
        args,
        augmented_state_dim,
        state_dim,
        action_dim,
        action_bound,
        device,
    )

    trainer = Trainer(env, eval_env, agent, random_seed, args)
    trainer.train()

if __name__ == "__main__":
    delay = 10
    arguments = get_parameters(delay)
    base_seed = arguments.random_seed
    for trial in range(arguments.num_trials):
        seed = base_seed + trial if base_seed > 0 else base_seed
        run_one(arguments, seed)
