import gymnasium as gym
from gymnasium import spaces

import random
import numpy as np
import yaml
import matplotlib.pyplot as plt

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

        self.action_stat = [0, 0]  # [do not put in production, put in production]

        # Define the space of actions and observations
        self.action_space = spaces.Discrete(2)  # 2 possible actions, put the order into production or not

        self.observation_space = spaces.Dict({
            "queue_lengths": spaces.MultiDiscrete([1000] * 6),  # List of integers
            "first_job_processing_times": spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32),  # List of floats
            "slack": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)  # Float
        })

    def step(self, action):
        # Sort the list by slack
        self.simpy_env.psp.sort(key=lambda job: job.dd - self.simpy_env.env.now)

        # If I put the job into production
        if action == 1:
            self.action_stat[1] += 1
            if len(self.simpy_env.psp) > 0:
                job = self.simpy_env.psp.pop(0)
                self.simpy_env.env.process(job.main())
            else:
                print("Nessun job da buttare dentro")
        else:
            self.action_stat[0] += 1

        # Wait for the next step
        time_step = rl_config['time_step_length']
        self.simpy_env.env.run(until=self.simpy_env.env.now + time_step)

        # Calculate the reward
        self.reward = self.simpy_env.get_reward()

        # Take the current observation
        obs = self.get_state()

        # Calculate if the episode is finished
        done = self.simpy_env.env.now >= 10005

        return obs, self.reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # TODO va bene?

        random.seed(seed)
        self.reward = 0
        self.action_stat = [0, 0]

        return self.get_state(), {}

    def render(self, mode='human'):
        # View current status
        queue_lengths = [len(machine.queue) for machine in self.simpy_env.machines]
        if len(self.simpy_env.psp) > 0:
            first_element = self.simpy_env.psp[0]

            print(
                f"Time: {self.simpy_env.env.now}, "
                f"queue_lengths: {queue_lengths}, "
                f"first_job_processing_times: {first_element.processing_times}, "
                f"slack: {first_element.dd - self.simpy_env.env.now}")
        else:
            print(
                f"Time: {self.simpy_env.env.now}, "
                f"queue_lengths: {queue_lengths}, "
                f"first_job_processing_times: [0., 0., 0., 0., 0., 0.], "
                f"slack: 0")

    def get_state(self):
        queue_lengths = np.array([len(machine.queue) for machine in self.simpy_env.machines])

        if len(self.simpy_env.psp) > 0:
            first_element = self.simpy_env.psp[0]

            return {
                "queue_lengths": queue_lengths,
                "first_job_processing_times": np.array(first_element.processing_times),
                "slack": np.array([first_element.dd - self.simpy_env.env.now])
            }
        return {
            "queue_lengths": queue_lengths,
            "first_job_processing_times": np.array([0., 0., 0., 0., 0., 0.]),
            "slack": np.array([0.])
        }


    def simpy_env_reset(self, sim_system: SimSystem) -> None:
        self.simpy_env = sim_system
