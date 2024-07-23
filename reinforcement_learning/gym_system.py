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

        if rl_config['normalize_state']:
            self.observation_space = spaces.Dict({
                "wip": spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32),  # List of floats
                "first_job_processing_times": spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32),  # List of floats
                "slack": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),  # Float
            })
        else:
            self.observation_space = spaces.Dict({
                "wip": spaces.Box(low=0, high=1000, shape=(6,), dtype=np.float32),  # List of floats
                "first_job_processing_times": spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32),
                # List of floats
                "slack": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),  # Float
            })

    @staticmethod
    def normalize_state(wip, first_job_processing_times, slack):
        """
        Normalizes the state variables for work-in-progress (WIP), first job processing times, and slack.
        This function scales the values of WIP, processing times of the first job, and slack to a normalized range
        between 0 and 1 based on predefined minimum and maximum values.
        :param wip: A list of work-in-progress values to be normalized.
        :param first_job_processing_times: A list of processing times for the first job to be normalized.
        :param slack: A list containing slack values to be normalized.
        :returns: A tuple containing normalized values for WIP, first job processing times, and slack.
        """

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
        normalized_slack = [0 if slack[0] < min_slack else 1 if slack[0] > max_slack else (slack[0] - min_slack) / (
                    max_slack - min_slack)]

        return normalized_wip, normalized_first_job_processing_times, normalized_slack

    @staticmethod
    def normalize_reward(reward: float) -> float:
        """
        Normalizes the reward value to a range between 0 and 1.
        :param reward: The reward value to be normalized.
        :returns: The normalized reward value.
        """

        reward_min = rl_config['normalizing']['min_reward']
        reward_max = rl_config['normalizing']['max_reward']

        return (reward - reward_min) / (reward_max - reward_min)

    def step(self, action):
        """
        Executes a step in the simulation environment based on the given action and updates the state.
        This function performs the following tasks:
        1. Records the action taken.
        2. If the action is to put a job into production (action == 1), it starts processing the first job in the queue.
        3. Sorts the jobs in the queue by their due date.
        4. Retrieves the current state observation.
        :param action: The action to be executed (0 or 1) representing whether to put a job into production or not.
        :returns: A tuple containing:
            - `obs`: The current state observation after taking the action.
            - `reward`: The normalized reward for the action taken.
            - `done`: A boolean indicating if the episode has finished.
            - `False`: A placeholder for additional information (not used in this implementation).
            - `{}`: An empty dictionary for additional information (not used in this implementation).
        """

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
        """
        Resets the simulation environment to its initial state.
        :param seed: Optional seed for random number generation to ensure reproducibility.
        :param options: Additional options for resetting the environment (currently not used).
        :returns: A tuple containing:
            - `self.get_state()`: The initial state of the environment after the reset.
            - `{}`: An empty dictionary for additional information (not used in this implementation).
        """

        super().reset(seed=seed)

        random.seed(seed)
        self.reward = 0
        self.action_stat = []
        self.not_job_to_push_action_0 = 0
        self.not_job_to_push_action_1 = 0

        return self.get_state(), {}

    def render(self, mode='human') -> None:
        """
        Displays the current status of the simulation environment.
        :param mode: The mode in which to render the output. Currently, only 'human' mode is supported, which prints
        the information to the console.
        :returns: None
        """

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
                f"slack: 0, "
            )

    def get_state(self):
        """
        Retrieves and returns the current state of the simulation environment.
        This function collects the current work-in-progress (WIP), the processing times of the first job in the queue,
        and the slack time (time remaining until the job's due date). It normalizes these values if normalization is
        enabled in the configuration.
        :returns: A dictionary containing the current state:
            - `wip`: A NumPy array representing the normalized work-in-progress values.
            - `first_job_processing_times`: A NumPy array representing the normalized processing times of the first job.
            - `slack`: A NumPy array representing the normalized slack time.
        """

        def get_raw_state():
            """
            Helper function to get the raw, unnormalized state of the environment.
            :returns: A tuple containing:
                - `wip`: The work-in-progress values.
                - `first_job_processing_times`: The processing times of the first job.
                - `slack`: The slack time (time remaining until the job's due date).
            """

            if len(self.simpy_env.psp) > 0:
                first_element = self.simpy_env.psp[0]
                first_job_processing_times = first_element.processing_times
                slack = [first_element.dd - self.simpy_env.env.now]
            else:
                first_job_processing_times = [0., 0., 0., 0., 0., 0.]
                slack = [0.]
            return self.simpy_env.get_wip(), first_job_processing_times, slack

        wip, first_job_processing_times, slack = get_raw_state()

        if rl_config['normalize_state']:
            wip, first_job_processing_times, slack = self.normalize_state(wip, first_job_processing_times, slack)

        return {
            "wip": np.array(wip),
            "first_job_processing_times": np.array(first_job_processing_times),
            "slack": np.array(slack),
        }

    def simpy_env_reset(self, sim_system: SimSystem) -> None:
        """
        Resets the simulation environment by setting a new simulation system.
        :param sim_system: The new simulation system to be assigned to `self.simpy_env`.
        :returns: None
        """

        self.simpy_env = sim_system
