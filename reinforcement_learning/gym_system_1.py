import gymnasium as gym
from gymnasium import spaces

import random
import numpy as np
import yaml

from simulation.sim_system import SimSystem


with open('../conf/sim_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

with open('../conf/rl_parameters.yaml', 'r') as rl_file:
    rl_config = yaml.safe_load(rl_file)


class GymSystem(gym.Env):
    def __init__(self, sim_system: SimSystem):
        super(GymSystem, self).__init__()
        self.simpy_env = sim_system

        self.reward = 0

        # To check
        self.action_stat: list[int] = []
        self.not_job_to_push_action_1 = 0
        self.not_job_to_push_action_0 = 0

        # Define the space of actions and observations
        self.action_space = spaces.Discrete(2)  # 2 possible actions, put the order into production or not

        self.observation_space = spaces.Dict({
            "wip": spaces.Box(low=0, high=1000, shape=(6,), dtype=np.float32),  # List of floats
            "first_job_processing_times": spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32),  # List of floats
            "slack": spaces.Discrete(100000)  # Float
        })

    def normalize_state(self, wip, first_job_processing_times, slack):
        wip_min = rl_config['normalizing']['min_wip']
        wip_max = rl_config['normalizing']['max_wip']
        min_processing_time = rl_config['normalizing']['min_processing_time']
        max_processing_time = rl_config['normalizing']['max_processing_time']
        min_slack = rl_config['normalizing']['min_slack']
        max_slack = rl_config['normalizing']['max_slack']

        # Normalize wip
        normalized_wip = [0 if x < wip_min else 1 if x > wip_max else (x - wip_min) / (wip_max - wip_min) for x in wip]

        # Normalize first_job_processing_times
        normalized_first_job_processing_times = [
            0 if x < min_processing_time else 1 if x > max_processing_time else (x - min_processing_time) / (
                        max_processing_time - min_processing_time) for x in first_job_processing_times
        ]

        # Normalize slack
        normalized_slack = [0 if slack < min_slack else 1 if slack > max_slack else (slack - min_slack) / (
                    max_slack - min_slack)]

        return normalized_wip, normalized_first_job_processing_times, normalized_slack

    def normalize_reward(self, reward):
        reward_min = rl_config['normalizing']['min_reward']
        reward_max = rl_config['normalizing']['max_reward']

        return (reward - reward_min) / (reward_max - reward_min)

    def step(self, action):
        self.action_stat.append(int(action))

        # If I put the job into production
        if action == 1:
            if len(self.simpy_env.psp) > 0:
                job = self.simpy_env.psp.pop(0)
                self.simpy_env.env.process(job.main())

        if action == 0 and len(self.simpy_env.psp) == 0:
            self.not_job_to_push_action_0 += 1
        elif action == 1 and len(self.simpy_env.psp) == 0:
            self.not_job_to_push_action_1 += 1

        # Wait for the next step
        time_step = rl_config['time_step_length']
        self.simpy_env.env.run(until=self.simpy_env.env.now + time_step)

        # Calculate the reward
        self.reward = self.simpy_env.get_reward()

        if rl_config['normalize_reward']:
            self.reward = self.normalize_reward(self.reward)

        # Sort the list by slack
        self.simpy_env.psp.sort(key=lambda job: job.dd - self.simpy_env.env.now)

        # Take the current observation
        obs = self.get_state()

        # Calculate if the episode is finished
        done = self.simpy_env.env.now >= rl_config['episode_length']

        return obs, self.reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        random.seed(seed)
        self.reward = 0
        self.action_stat = []
        self.not_job_to_push_action_0 = 0
        self.not_job_to_push_action_1 = 0

        return self.get_state(), {}

    def render(self, mode='human'):
        # View current status
        if len(self.simpy_env.psp) > 0:
            first_element = self.simpy_env.psp[0]

            print(
                f"Time: {self.simpy_env.env.now}, "
                f"wip: {self.simpy_env.get_wip()}, "                
                f"first_job_processing_times: {first_element.processing_times}, "
                f"slack: {first_element.dd - self.simpy_env.env.now}, "
            )
        else:
            print(
                f"Time: {self.simpy_env.env.now}, "
                f"wip: {self.simpy_env.get_wip()}, "
                f"first_job_processing_times: [0., 0., 0., 0., 0., 0.], "
                f"slack: 0"
            )

    def get_state(self):
        def get_raw_state():
            if len(self.simpy_env.psp) > 0:
                first_element = self.simpy_env.psp[0]
                first_job_processing_times = first_element.processing_times
                slack = first_element.dd - self.simpy_env.env.now
            else:
                first_job_processing_times = [0., 0., 0., 0., 0., 0.]
                slack = 0
            return self.simpy_env.get_wip(), first_job_processing_times, slack

        wip, first_job_processing_times, slack = get_raw_state()

        if rl_config['normalize_state']:
            wip, first_job_processing_times, slack = self.normalize_state(wip, first_job_processing_times, slack)

        return {
            "wip": np.array(wip),
            "first_job_processing_times": np.array(first_job_processing_times),
            "slack": slack,
        }

    def simpy_env_reset(self, sim_system: SimSystem) -> None:
        self.simpy_env = sim_system
