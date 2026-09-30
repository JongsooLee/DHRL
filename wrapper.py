from collections import deque

import gym
import numpy as np


class DelayedEnv(gym.Wrapper):
    def __init__(self, env, seed, obs_delayed_steps, act_delayed_steps):
        super().__init__(env)
        assert obs_delayed_steps + act_delayed_steps > 0

        self.seed_value = int(seed)
        self._first_reset = True
        self.env.action_space.seed(self.seed_value)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._max_episode_steps = self.env._max_episode_steps

        self.obs_buffer = deque(maxlen=obs_delayed_steps)
        self.reward_buffer = deque(maxlen=obs_delayed_steps)
        self.done_buffer = deque(maxlen=obs_delayed_steps)
        self.action_buffer = deque(maxlen=act_delayed_steps)

        self.obs_delayed_steps = int(obs_delayed_steps)
        self.act_delayed_steps = int(act_delayed_steps)

    def reset(self):

        self.obs_buffer.clear()
        self.reward_buffer.clear()
        self.done_buffer.clear()
        self.action_buffer.clear()

        if self._first_reset:
            reset_result = self.env.reset(seed=self.seed_value)
            self._first_reset = False
        else:
            reset_result = self.env.reset()

        init_state = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        for _ in range(self.act_delayed_steps):
            self.action_buffer.append(np.zeros(self.action_space.shape, dtype=self.action_space.dtype))

        for _ in range(self.obs_delayed_steps):
            self.obs_buffer.append(np.asarray(init_state).copy())
            self.reward_buffer.append(0.0)
            self.done_buffer.append(False)
        return init_state

    def step(self, action):
        if self.act_delayed_steps > 0:
            delayed_action = self.action_buffer.popleft()
            self.action_buffer.append(np.asarray(action).copy())
        else:
            delayed_action = action

        step_result = self.env.step(delayed_action)

        if len(step_result) == 5:
            current_obs, current_reward, terminated, truncated, info = step_result
            current_done = bool(terminated or truncated)
        else:
            current_obs, current_reward, current_done, info = step_result
            current_done = bool(current_done)

        if self.obs_delayed_steps > 0:
            delayed_obs = self.obs_buffer.popleft()
            delayed_reward = self.reward_buffer.popleft()
            delayed_done = self.done_buffer.popleft()

            self.obs_buffer.append(np.asarray(current_obs).copy())
            self.reward_buffer.append(float(current_reward))
            self.done_buffer.append(current_done)
        else:
            delayed_obs = current_obs
            delayed_reward = current_reward
            delayed_done = current_done

        output_info = dict(info) if isinstance(info, dict) else {}
        output_info.update(
            {
                "current_obs": current_obs,
                "current_reward": current_reward,
                "current_done": current_done,
            }
        )
        return delayed_obs, delayed_reward, delayed_done, output_info
