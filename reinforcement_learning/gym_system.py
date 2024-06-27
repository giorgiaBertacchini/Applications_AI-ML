import gymnasium as gym
from gymnasium import spaces

import random
import numpy as np
import yaml
import matplotlib.pyplot as plt

from simulation.sim_system import SimSystem
from simulation.job import Job


with open('../conf/sim_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

with open('../conf/rl_parameters.yaml', 'r') as rl_file:
    rl_config = yaml.safe_load(rl_file)


class GymSystem(gym.Env):
    def __init__(self, sim_system: SimSystem):
        super(GymSystem, self).__init__()
        self.simpy_env = sim_system

        self.reward = 0
        self.psp: list[Job] = []  # “pre-shop pool” (PSP)

        self.rewards_over_time = []

        # Definire lo spazio delle azioni e delle osservazioni
        self.action_space = spaces.Discrete(2)  # Esempio: 2 azioni possibili, metti in produczione o no l'ordine

        self.observation_space = spaces.Dict({
            "queue_lengths": spaces.MultiDiscrete([1000] * 6),  # Lista di interi
            "first_job_routing": spaces.MultiBinary(6),  # Lista di booleani
            "slack": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)  # Float
        })

    def step(self, action):
        # Eseguire l'azione, butto dentro il job
        if action == 1:
            if len(self.simpy_env.psp) > 0:
                job = self.simpy_env.psp.pop()
                self.simpy_env.env.process(job.main())  # TODO va bene? perdiamo tempo li?
                self.job_penalty(job)
            else:
                print("Nessun job da buttare dentro")

        # Wait for the next step
        time_step = rl_config['time_step_length']
        self.simpy_env.env.run(until=self.simpy_env.env.now + time_step)

        # Calcolare la ricompensa
        self.recurrent_penalty()

        # Calcolare l'osservazione
        obs = self.get_state()

        # Calcolare se l'episodio è finito
        done = self.simpy_env.env.now >= 10000

        return obs, self.reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # TODO va bene?
        # creo un nuovo simpy_env
        random.seed(seed)
        #self.simpy_env = SimSystem(..)

        self.reward = 0
        self.rewards_over_time = []

        #self.simpy_env.env.process(self.simpy_env.run())  # TODO va bene?

        return self.get_state(), {}

    def render(self, mode='human'):
        # Visualizza lo stato attuale
        queue_lengths = [len(machine.queue) for machine in self.simpy_env.machines]
        if len(self.psp) > 0:
            first_element = self.psp[0]

            print(
                f"Time: {self.simpy_env.env.now}, "
                f"queue_lengths: {queue_lengths}, "
                f"first_job_routing: {first_element.routing}, "
                f"slack: {first_element.dd - self.simpy_env.env.now}")
        else:
            print(
                f"Time: {self.simpy_env.env.now}, "
                f"queue_lengths: {queue_lengths}, "
                f"first_job_routing: (0, 0, 0, 0, 0, 0), "
                f"slack: 0")

    def get_state(self):
        queue_lengths = [len(machine.queue) for machine in self.simpy_env.machines]

        if len(self.psp) > 0:
            first_element = self.psp[0]

            return {
                "queue_lengths": queue_lengths,
                "first_job_routing": list(first_element.routing),
                "slack": [first_element.dd - self.simpy_env.env.now]
            }
        return {
            "queue_lengths": queue_lengths,
            "first_job_routing": [0, 0, 0, 0, 0, 0],  # nel caso non ci siano nuovi job?
            "slack": [0.]
        }

    def job_penalty(self, job) -> None:
        # penalty solo quando terminato l'item.
        # Ma potrei anche al giorno vedere il mio penalty aumentare nel caso di ritardi

        delivery_window = config['delivery_window']
        daily_penalty = config['daily_penalty']

        if job.dd - delivery_window > self.simpy_env.env.now:
            # troppo in anticipo
            self.reward -= daily_penalty * (job.dd - self.simpy_env.env.now - delivery_window)

        #if job.dd < (self.simpy_env.env.now - delivery_window):
        #    # troppo in ritardo
        #    self.reward -= daily_penalty * (job.dd - (self.simpy_env.env.now - delivery_window))

    def recurrent_penalty(self) -> None:
        # penalità ricorrente
        # per ogni unità
        time_step_length = rl_config['time_step_length']
        delivery_window = config['delivery_window']
        daily_penalty = config['daily_penalty']

        for machine in self.simpy_env.machines:
            for job in machine.queue:
                # nel caso di ritardi
                if job.dd + delivery_window < self.simpy_env.env.now:
                    delay = self.simpy_env.env.now - job.dd - delivery_window

                    # Se il ritardo è inferiore al time_step
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
