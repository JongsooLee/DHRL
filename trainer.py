import time
import numpy as np
from utils import initialize_log_file, log_to_txt, write_run_summary


class Trainer:
    def __init__(self, env, eval_env, agent, random_seed, args):
        self.args = args
        self.agent = agent
        self.random_seed = int(random_seed)

        self.delayed_env = env
        self.eval_delayed_env = eval_env

        self.start_step = args.start_step
        self.update_after = args.update_after
        self.max_step = args.max_step
        self.batch_size = args.batch_size
        self.update_every = args.update_every

        self.eval_flag = args.eval_flag
        self.eval_episode = args.eval_episode
        self.eval_freq = args.eval_freq

        self.episode = 0
        self.total_step = 0
        self.local_step = 0
        self.eval_local_step = 0
        self.finish_flag = False
        self.gradient_updates = 0

        self.total_delayed_steps = (
            args.obs_delayed_steps + args.act_delayed_steps
        )

        initialize_log_file(args, self.random_seed)

    @staticmethod
    def _zero_action(env):
        return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    def train(self):
        start_time = time.perf_counter()
        try:
            while not self.finish_flag:
                self.episode += 1
                self.local_step = 0

                self.delayed_env.reset()
                self.agent.temporary_buffer.clear()
                done = False

                while not done and not self.finish_flag:
                    self.local_step += 1
                    self.total_step += 1

                    if self.local_step < self.total_delayed_steps:
                        action = self._zero_action(self.delayed_env)
                        self.delayed_env.step(action)
                        self.agent.temporary_buffer.actions.append(action)

                    elif self.local_step == self.total_delayed_steps:
                        if self.total_step < self.start_step:
                            action = self.delayed_env.action_space.sample()
                        else:
                            action = self._zero_action(self.delayed_env)

                        next_observed_state, _, _, _ = self.delayed_env.step(action)
                        self.agent.temporary_buffer.actions.append(action)
                        self.agent.temporary_buffer.states.append(next_observed_state)

                    else:
                        last_observed_state = self.agent.temporary_buffer.states[-1]
                        first_action_idx = (len(self.agent.temporary_buffer.actions) - self.total_delayed_steps)
                        augmented_state = (self.agent.temporary_buffer.get_augmented_state(last_observed_state, first_action_idx))

                        if self.total_step < self.start_step:
                            action = self.delayed_env.action_space.sample()
                        else:
                            action = self.agent.get_action(augmented_state, evaluation=False)

                        next_observed_state, reward, done, _ = (self.delayed_env.step(action))
                        true_done = 0.0 if self.local_step == self.delayed_env._max_episode_steps + self.args.obs_delayed_steps else float(done)

                        self.agent.temporary_buffer.actions.append(action)
                        self.agent.temporary_buffer.states.append(next_observed_state)

                        if self.local_step > 2 * self.total_delayed_steps:
                            augmented_s, state, replay_action, next_augmented_s, next_state = self.agent.temporary_buffer.get_tuple()
                            self.agent.buffer.push(augmented_s, state, replay_action, reward, next_augmented_s, next_state, true_done)

                    if self.agent.buffer.size >= self.batch_size \
                            and self.total_step >= self.update_after \
                            and self.total_step % self.update_every == 0:
                        for update_index in range(self.update_every):
                            self.agent.train(self.total_step, update_index)
                            self.gradient_updates += 1

                    if self.eval_flag and self.total_step % self.eval_freq == 0:
                        self.evaluate()

                    if self.total_step >= self.max_step:
                        self.finish_flag = True

        finally:
            elapsed = time.perf_counter() - start_time
            write_run_summary(
                self.args,
                self.random_seed,
                {
                    "environment_steps": int(self.total_step),
                    "gradient_updates": int(self.gradient_updates),
                    "episodes": int(self.episode),
                    "wall_clock_seconds": float(elapsed),
                },
            )
            self.delayed_env.close()
            self.eval_delayed_env.close()

    def evaluate(self):
        reward_list = []

        for _ in range(self.eval_episode):
            episode_reward = 0.0
            self.eval_delayed_env.reset()
            self.agent.eval_temporary_buffer.clear()
            done = False
            self.eval_local_step = 0

            while not done:
                self.eval_local_step += 1
                if self.eval_local_step < self.total_delayed_steps:
                    action = self._zero_action(self.eval_delayed_env)
                    self.eval_delayed_env.step(action)
                    self.agent.eval_temporary_buffer.actions.append(action)

                elif self.eval_local_step == self.total_delayed_steps:
                    action = self._zero_action(self.eval_delayed_env)
                    next_observed_state, _, _, _ = self.eval_delayed_env.step(action)

                    self.agent.eval_temporary_buffer.actions.append(action)
                    self.agent.eval_temporary_buffer.states.append(next_observed_state)


                else:
                    last_observed_state = self.agent.eval_temporary_buffer.states[-1]
                    first_action_idx = len(self.agent.eval_temporary_buffer.actions) - self.total_delayed_steps
                    augmented_state = self.agent.eval_temporary_buffer.get_augmented_state(last_observed_state, first_action_idx)

                    action = self.agent.get_action(augmented_state, evaluation=True)
                    next_observed_state, reward, done, _ = self.eval_delayed_env.step(action)
                    self.agent.eval_temporary_buffer.actions.append(action)
                    self.agent.eval_temporary_buffer.states.append(next_observed_state)
                    episode_reward += reward

            reward_list.append(episode_reward)

        average_reward = float(np.mean(reward_list))
        log_to_txt(self.args, self.total_step, average_reward, self.random_seed)
        print(
            "Eval  |  Total Steps {}  |  Episodes {}  |  "
            "Average Reward {:.2f}  |  Max reward {:.2f}  |  "
            "Min reward {:.2f}".format(
                self.total_step,
                self.episode,
                average_reward,
                max(reward_list),
                min(reward_list),
            )
        )
