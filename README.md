### Delayed Homomorphic Reinforcement Learning for Environments with Delayed Feedback

[![Python Badge](https://img.shields.io/badge/Python-3.8-blue?logo=python&style=flat-square)](https://www.python.org/)
[![PyTorch Badge](https://img.shields.io/badge/PyTorch-2.0.0-%23EE4C2C?logo=pytorch&style=flat-square)](https://pytorch.org/)
[![NeurIPS 2026 Badge](https://img.shields.io/badge/NeurIPS%202026-Paper-%23007ACC?style=flat-square)](https://arxiv.org/abs/2604.03641)

![NeurIPS Logo](figures/neurips_logo.png)

> PyTorch implementation of Deep Delayed Homomorphic Policy Gradient  
> Paper link: https://arxiv.org/abs/2604.03641

---

#### Project structure

    .
    ├── main.py                  # Entry point & arguments
    ├── trainer.py               # Training / evaluation loop in delayed environments
    ├── wrapper.py               # Delayed environment wrapper (observation / action delay)
    ├── core.py                  # Actor, critic, homomorphism map, abstract models
    ├── replay_memory.py         # Replay buffer (augmented & time-aligned transitions)
    ├── temporary_buffer.py      # Per-episode buffer to build augmented states
    ├── utils.py                 # Seeding, logging, etc
    └── agents/
        ├── d2hpg_bpql.py        # D2HPG-BPQL  (BPQL as the regular-policy learner)
        ├── d2hpg_sac.py         # D2HPG-SAC   (SAC as the regular-policy learner)
        ├── d2hpg_naive.py       # D2HPG-naive (use homomorphic policy gradient only)
        └── d2hpg_principled.py  # Principled D2HPG (learns the homomorphism components)


---

#### Test environments

    python == 3.8.0  
    pytorch == 2.0.0  
    mujoco == 2.2.0  
    mujoco_py == 2.1.2.14  
    gym == 0.26.2  
    
---

#### Runs

> python main.py --env-name HalfCheetah-v3 --obs-delayed-steps 10 --agent-type bpql --random-seed 1 --num-trials 5 --max-step 1000000

---

#### Acknowledgement

> [Belief Projection-based Q-learning](https://github.com/jangwonkim-cocel/BPQL)  
> [Deep Homomorphic Policy Gradient](https://github.com/sahandrez/homomorphic_policy_gradient)  
