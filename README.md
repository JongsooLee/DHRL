<h3 align="center">Delayed Homomorphic Reinforcement Learning for Environments with Delayed Feedback</h3>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8-blue?logo=python&style=flat-square" alt="Python Badge"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0.0-%23EE4C2C?logo=pytorch&style=flat-square" alt="PyTorch Badge"></a>
  <a href="https://arxiv.org/abs/2604.03641"><img src="https://img.shields.io/badge/NeurIPS%202026-Paper-%23007ACC?style=flat-square" alt="NeurIPS 2026 Badge"></a>
</p>

<p align="center">
  <img src="figures/neurips_logo.png" alt="NeurIPS Logo" width="300">
</p>

#### Overview

Reinforcement learning in real-world systems often involves delayed feedback, which violates the Markov assumption and impedes both learning and control. Canonical augmentation-based approaches address this issue by augmenting the state with action histories, but suffer from state-space explosion and the resulting sample-complexity burden. Delayed Homomorphic Reinforcement Learning (DHRL) is a framework grounded in MDP homomorphisms that identifies and collapses control-redundant augmented states into a compact abstract state space, providing a unified abstraction mechanism for both the actor and critic. Deep Delayed Homomorphic Policy Gradient (D²HPG) is a deep actor-critic instantiation of the DHRL framework for continuous domains.

---

> PyTorch implementation of deep delayed homomorphic policy gradient algorithm  
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

    python main.py --env-name HalfCheetah-v3 --obs-delayed-steps 10 --agent-type bpql --random-seed 1 --num-trials 5 --max-step 1000000

---


#### Citation

    @article{lee2026delayed,
      title={Delayed homomorphic reinforcement learning for environments with delayed feedback},
      author={Lee, Jongsoo and Kim, Jangwon and Han, Soohee},
      journal={arXiv preprint arXiv:2604.03641},
      year={2026}
    }
