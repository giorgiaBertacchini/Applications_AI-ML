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


class GymExpandedSystem(gym.Env):
    def __init__(self, sim_system: SimSystem):
        super(GymExpandedSystem, self).__init__()
        self.simpy_env = sim_system

        self.reward = 0
        #self.psp: list[Job] = []  # “pre-shop pool” (PSP)

        self.rewards_over_time = []

        # Definire lo spazio delle azioni e delle osservazioni
        self.action_space = spaces.Discrete(2)  # Esempio: 2 azioni possibili, metti in produzione o no l'ordine

        self.observation_space = spaces.Dict({
            "queue_lengths": spaces.MultiDiscrete([1000] * 6),  # Lista di interi
            "first_job_processing_times": spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32),  # List of floats
            "slack": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),  # Float
            "psp_length": spaces.Discrete(1000000),  # Lunghezza della lista psp
        })

    def step(self, action):
        # TODO riordinare la lista
        self.simpy_env.psp.sort(key=lambda job: job.dd - self.simpy_env.env.now)

        # Eseguire l'azione, butto dentro il job
        if action == 1:
            if len(self.simpy_env.psp) > 0:
                job = self.simpy_env.psp.pop(0)
                self.simpy_env.env.process(job.main())
            else:
                print("Nessun job da buttare dentro")

        # Wait for the next step
        time_step = rl_config['time_step_length']
        self.simpy_env.env.run(until=self.simpy_env.env.now + time_step)

        # Calculate the reward
        self.simpy_env.get_reward()

        # Calcolare l'osservazione
        obs = self.get_state()

        # Calcolare se l'episodio è finito
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
        # Visualizza lo stato attuale
        queue_lengths = [len(machine.queue) for machine in self.simpy_env.machines]
        if len(self.simpy_env.psp) > 0:
            first_element = self.simpy_env.psp[0]

            processing_times = np.array(first_element.processing_times)
            padded_processing_times = np.pad(processing_times, (0, 6 - len(processing_times)))

            print(
                f"Time: {self.simpy_env.env.now}, "
                f"queue_lengths: {queue_lengths}, "
                f"first_job_processing_times: {first_element.processing_times}, "
                f"slack: {first_element.dd - self.simpy_env.env.now}, "
                f"psp_length: {len(self.simpy_env.psp)} ")
        else:
            print(
                f"Time: {self.simpy_env.env.now}, "
                f"queue_lengths: {queue_lengths}, "
                f"first_job_processing_times: [0., 0., 0., 0., 0., 0.], "
                f"slack: 0, "
                f"psp_length: 0")

    def get_state(self):
        queue_lengths = np.array([len(machine.queue) for machine in self.simpy_env.machines])

        if len(self.simpy_env.psp) > 0:
            first_element = self.simpy_env.psp[0]

            processing_times = np.array(first_element.processing_times)
            padded_processing_times = np.pad(processing_times, (0, 6 - len(processing_times)))

            return {
                "queue_lengths": queue_lengths,
                "first_job_processing_times": np.array(first_element.processing_times),
                "slack": np.array([first_element.dd - self.simpy_env.env.now]),
                "psp_length": len(self.simpy_env.psp)
            }
        return {
            "queue_lengths": queue_lengths,
            "first_job_processing_times": np.array([0., 0., 0., 0., 0., 0.]),
            "slack": np.array([0.]),
            "psp_length": 0
        }

    def plot_rewards_over_time(self):
        plt.plot(self.rewards_over_time)
        plt.xlabel('Time step')
        plt.ylabel('Reward')
        plt.title('Rewards Over Time')
        plt.show()

    def simpy_env_reset(self, sim_system: SimSystem) -> None:
        self.simpy_env = sim_system
