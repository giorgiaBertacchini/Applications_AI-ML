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
        self.rewards_over_time = []

        # Define the space of actions and observations
        self.action_space = spaces.Discrete(2)  # 2 possible actions, put the order into production or not

        self.observation_space = spaces.Dict({
            "queue_lengths": spaces.MultiDiscrete([1000] * 6),  # List of integers
            "first_job_routing": spaces.MultiBinary(6),  # List of bools
            "slack": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)  # Float
        })

    def step(self, action):
        # Sort the list by slack
        self.simpy_env.psp.sort(key=lambda job: job.dd - self.simpy_env.env.now)

        # If I put the job into production
        if action == 1:
            if len(self.simpy_env.psp) > 0:
                job = self.simpy_env.psp.pop(0)
                self.simpy_env.env.process(job.main())
                self.job_penalty(job)
            else:
                print("Nessun job da buttare dentro")

        # Wait for the next step
        time_step = rl_config['time_step_length']
        self.simpy_env.env.run(until=self.simpy_env.env.now + time_step)

        # Calculate the reward
        self.recurrent_penalty()

        # Take the current observation
        obs = self.get_state()

        # Calculate if the episode is finished
        done = self.simpy_env.env.now >= 10005

        #if done:
        #    self.plot_rewards_over_time()

        return obs, self.reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # TODO va bene?

        random.seed(seed)
        self.reward = 0
        self.rewards_over_time = []

        return self.get_state(), {}

    def render(self, mode='human'):
        # View current status
        queue_lengths = [len(machine.queue) for machine in self.simpy_env.machines]
        if len(self.simpy_env.psp) > 0:
            first_element = self.simpy_env.psp[0]

            print(
                f"Time: {self.simpy_env.env.now}, "
                f"queue_lengths: {queue_lengths}, "
                f"first_job_routing: {first_element.routing}, "
                f"slack: {first_element.dd - self.simpy_env.env.now}")
        else:
            print(
                f"Time: {self.simpy_env.env.now}, "
                f"queue_lengths: {queue_lengths}, "
                f"first_job_routing: (False, False, False, False, False, False), "
                f"slack: 0")

    def get_state(self):
        queue_lengths = np.array([len(machine.queue) for machine in self.simpy_env.machines])  # TODO non c'è np.array

        if len(self.simpy_env.psp) > 0:
            first_element = self.simpy_env.psp[0]

            return {
                "queue_lengths": queue_lengths,
                "first_job_routing": np.array(first_element.routing),  # TODO era list()
                "slack": np.array([first_element.dd - self.simpy_env.env.now])
            }
        return {
            "queue_lengths": queue_lengths,
            "first_job_routing": np.array([False, False, False, False, False, False]),  # TODO era senza np.array
            "slack": np.array([0.])
        }

    def job_penalty(self, job) -> None:
        # When the job is completed
        delivery_window = rl_config['delivery_window']
        daily_penalty = rl_config['daily_penalty']

        if job.dd - delivery_window > self.simpy_env.env.now:
            # too early
            self.reward -= daily_penalty * (job.dd - self.simpy_env.env.now - delivery_window)

        #if job.dd < (self.simpy_env.env.now - delivery_window):
        #    # troppo in ritardo
        #    self.reward -= daily_penalty * (job.dd - (self.simpy_env.env.now - delivery_window))

    def recurrent_penalty(self) -> None:
        time_step_length = rl_config['time_step_length']
        delivery_window = rl_config['delivery_window']
        daily_penalty = rl_config['daily_penalty']

        for machine in self.simpy_env.machines:
            for req in machine.queue:
                # in case of delays
                if req.job.dd + delivery_window < self.simpy_env.env.now:
                    delay = self.simpy_env.env.now - req.job.dd - delivery_window

                    # If the delay is less than the time_step
                    if delay / time_step_length < 1:
                        self.reward -= daily_penalty * delay
                    else:
                        self.reward -= daily_penalty * time_step_length

        self.rewards_over_time.append(self.reward)

    def plot_rewards_over_time(self):
        plt.plot(self.rewards_over_time)
        plt.xlabel('Time step')
        plt.ylabel('Reward')
        plt.title('Rewards Over Time')
        plt.show()

    def simpy_env_reset(self, sim_system: SimSystem) -> None:
        self.simpy_env = sim_system
