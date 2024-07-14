import simpy
import yaml
import statistics

from typing import List  # TODO va bene?
from collections.abc import Callable, Generator

from simulation.machine import Machine
from simulation.job import Job

with open('../conf/sim_config.yaml', 'r') as file:
    config = yaml.safe_load(file)


class SimSystem:
    def __init__(
            self,
            env: simpy.Environment,
            inter_arrival_time_distribution: Callable[[], float],
            family_index: Callable[[], int],
            f1_processing_time_distribution: Callable[[], float],
            f2_processing_time_distribution: Callable[[], float],
            f3_processing_time_distribution: Callable[[], float],
            routing: Callable[[], List[float]],
            dd: Callable[[], float]
    ):
        self.env = env
        self.inter_arrival_time_distribution = inter_arrival_time_distribution
        self.family_index = family_index
        self.f1_processing_time_distribution = f1_processing_time_distribution
        self.f2_processing_time_distribution = f2_processing_time_distribution
        self.f3_processing_time_distribution = f3_processing_time_distribution
        self.routing = routing
        self.dd = dd

        self.machines: list[Machine] = []
        self.jobs: list[Job] = []
        self.jobs_inter_arrival_times: list[float] = []

        for i in range(6):
            self.machines.append(Machine(
                env=self.env,
                capacity=1
            ))

        self.psp: list[Job] = []  # “pre-shop pool” (PSP)

        # Statistics: throughput
        self.th_stats: list[int] = [0]
        self.last_total_th = 0

        # Statistics: wip
        self.wip_stats: list[tuple[int]] = [(0, 0, 0, 0, 0, 0)]

        # Statistics: mean time in system
        self.mts_stats: list[float] = []

        # Statistics: mean delay in system
        self.mds_stats: list[float] = []

        # Statistics: reward
        self.rewards_over_time: list[float] = []
        self.reward_time_step_jobs: list[Job] = []
        self.last_queue_lengths: list[int] = [0, 0, 0, 0, 0, 0]

        # To normalize
        self.max_wip = float('-inf')
        self.max_first_job_processing_times = float('-inf')

        self.env.process(self.run())

        if config['throughput_sampling']:
            self.env.process(self.throughput_sampler())

        if config['wip_sampling']:
            self.env.process(self.wip_sampler())

        if config['mean_time_in_system_sampling']:
            self.env.process(self.mean_time_in_system_sampler())

        if config['mean_delay_in_system_sampling']:
            self.env.process(self.mean_delay_in_system_sampler())

        if config['reward']['reward_sampling']:
            self.env.process(self.reward_sampler())

    @property
    def finished_jobs(self) -> int:
        return sum(job.done for job in self.jobs)

    @property
    def correctly_finished_jobs(self) -> int:
        half_window = config['reward']['delivery_half_window']
        count = 0
        for job in self.jobs:
            if job.done:
                # Assuming job.env.now stores the delivery time when the job was marked as done
                # and job.dd is the due date for the job
                if job.dd - half_window <= job.delivery_time <= job.dd + half_window:
                    count += 1
        return count

    def get_wip(self) -> List[float]:
        wip = []
        for i in range(6):
            machine_wip = 0
            for j, machine in enumerate(self.machines):
                if j <= i:
                    for job_request in machine.queue:
                        machine_wip += job_request.job.processing_times[i]
            wip.append(machine_wip)

        return wip

    def wip_sampler(self) -> Generator[simpy.Event, None, None]:
        while True:
            yield self.env.timeout(float(config['wip_timestep']))
            wip = self.get_wip()
            if max(wip) > self.max_wip:
                self.max_wip = max(wip)
            self.wip_stats.append(tuple(wip))

    def throughput_sampler(self) -> Generator[simpy.Event, None, None]:
        while True:
            yield self.env.timeout(float(config['throughput_timestep']))
            delta = self.finished_jobs - self.last_total_th
            self.th_stats.append(delta)
            self.last_total_th = self.finished_jobs

    def mean_time_in_system_sampler(self) -> Generator[simpy.Event, None, None]:
        while True:
            yield self.env.timeout(float(config['mean_time_in_system_timestep']))

            mean_mts = statistics.mean(
                sum(job.delays) + sum(p for _, p in job.real_routing) for job in self.jobs)  # TODO si può migliorare
            self.mts_stats.append(mean_mts)

    def mean_delay_in_system_sampler(self) -> Generator[simpy.Event, None, None]:  # devo fare la media dei tempi di attesa in coda e di servizio
        while True:
            yield self.env.timeout(float(config['mean_delay_in_system_timestep']))
            total_delay = []
            for job in self.jobs:
                total_delay.append(sum(job.delays))
            if len(total_delay) > 0:
                mean_mds = statistics.mean(total_delay)
                self.mds_stats.append(mean_mds)
            else:
                self.mds_stats.append(0)
            #mean_mds = statistics.mean(sum(job.delays) for job in self.jobs)
            #self.mds_stats.append(mean_mds)

    def get_reward(self) -> float:
        delivery_half_window = config['reward']['delivery_half_window']
        daily_penalty = config['reward']['daily_penalty']
        award_delivery = config['reward']['award_delivery']
        wip_penalty = config['reward']['wip_penalty']
        wip_award = config['reward']['wip_award']
        avoid_empty_queue_penalty = config['reward']['avoid_empty_queue_penalty']
        job_late_penalty = config['reward']['job_late_penalty']

        reward: float = 0.0

        for job in self.reward_time_step_jobs.copy():
            if job.done:
                reward += award_delivery
                if job.dd - delivery_half_window > self.env.now:
                    # too early
                    reward -= daily_penalty * (job.dd - self.env.now - delivery_half_window)

                elif job.dd + delivery_half_window < self.env.now:
                    # too late
                    reward -= daily_penalty * (self.env.now - job.dd - delivery_half_window)

                # Remove the job from the original list
                self.reward_time_step_jobs.remove(job)


        actual_wip = self.get_wip()
        for i, machine in enumerate(self.machines):
            queue_length_difference = actual_wip[i] - self.last_queue_lengths[i]
            if queue_length_difference < 0:
                reward += (wip_award * queue_length_difference)
            elif queue_length_difference > 0:
                reward -= (wip_penalty * queue_length_difference)
        '''
        for i, machine in enumerate(self.machines):
            queue_length_difference = len(machine.queue) - self.last_queue_lengths[i]
            if queue_length_difference < 0:  # The queue has decreased
                reward += (wip_award * queue_length_difference)  # Increase the reward
            elif queue_length_difference > 0:  # The queue has increased
                reward -= (wip_penalty * queue_length_difference) # Decrease the reward
        '''
        if all(value == 0 for value in self.get_wip()) and len(self.psp) > 0:
            reward -= avoid_empty_queue_penalty  # TODO da valutare, per evitare che il sistema blocchi i processi e non consegni più

        # Update the last queue lengths for the next timestep
        self.last_queue_lengths = actual_wip
        #self.last_queue_lengths = [len(machine.queue) for machine in self.machines]

        if len(self.psp) > 0:
            if self.psp[0].dd - self.env.now < 0:
                # If the job to evaluate is already late
                self.get_wip()
                #reward -= job_late_penalty
                processing_times = self.psp[0].processing_times
                wip_values = self.get_wip()

                # Somma i valori di processing_times e wip_values a pari indici dove processing_times > 0
                summed_values = [processing_time + wip_value if processing_time > 0 else wip_value
                                 for processing_time, wip_value in zip(processing_times, wip_values)]

                reward -= ((-(self.psp[0].dd - self.env.now) + sum(summed_values)) * daily_penalty)

        return reward

    def reward_sampler(self) -> Generator[simpy.Event, None, None]:
        while True:
            yield self.env.timeout(float(config['reward']['reward_time_step']))

            self.rewards_over_time.append(self.get_reward())

    def run(self) -> Generator[simpy.Event, None, None]:
        while True:
            jobs_inter_arrival_time = self.inter_arrival_time_distribution()
            self.jobs_inter_arrival_times.append(jobs_inter_arrival_time)

            # Wait for the next customer to arrive
            yield self.env.timeout(jobs_inter_arrival_time)

            # Create a new job
            family_group = self.family_index()

            config_routing = config['families']['f{}'.format(family_group)]['routing']
            family_routing = self.routing()
            new_routing = [True if family_routing[i] <= config_routing[i] else False for i in range(6)]

            processing_time_list = []
            for i in range(6):
                if new_routing[i]:
                    if family_group == 1:
                        processing_time_list.append(self.f1_processing_time_distribution())
                    elif family_group == 2:
                        processing_time_list.append(self.f2_processing_time_distribution())
                    else:
                        processing_time_list.append(self.f3_processing_time_distribution())
                else:
                    processing_time_list.append(0)

            job = Job(
                env=self.env,
                family_group=family_group,
                processing_times=processing_time_list,
                machines=self.machines,
                dd=self.dd() + self.env.now
            )

            if max(processing_time_list) > self.max_first_job_processing_times:
                self.max_first_job_processing_times = max(processing_time_list)

            if config['reward']['reward_sampling']:
                self.reward_time_step_jobs.append(job)

            self.job_manager(job)

    def job_manager(self, job: Job) -> None:
        self.jobs.append(job)
        self.psp.append(job)
