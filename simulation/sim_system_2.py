import simpy
import yaml
import statistics

from typing import List  # TODO va bene?
from collections.abc import Callable, Generator

from simulation.machine import Machine
from simulation.job_2 import Job

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

        self.env.process(self.run())

        if config['throughput_sampling']:
            self.env.process(self.throughput_sampler())

        if config['wip_sampling']:
            self.env.process(self.wip_sampler())

        if config['mean_time_in_system_sampling']:
            self.env.process(self.mean_time_in_system_sampler())

        if config['mean_delay_in_system_sampling']:
            self.env.process(self.mean_delay_in_system_sampler())

    @property
    def finished_jobs(self) -> int:
        return sum(job.done for job in self.jobs)

    def wip_sampler(self) -> Generator[simpy.Event, None, None]:
        while True:
            yield self.env.timeout(float(config['wip_timestep']))
            wip = tuple(machine.queue_length for machine in self.machines)
            self.wip_stats.append(wip)

    def throughput_sampler(self) -> Generator[simpy.Event, None, None]:
        while True:
            yield self.env.timeout(float(config['throughput_timestep']))
            delta = self.finished_jobs - self.last_total_th
            self.th_stats.append(delta)
            self.last_total_th = self.finished_jobs

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

    def mean_time_in_system_sampler(self) -> Generator[simpy.Event, None, None]:
        while True:
            yield self.env.timeout(float(config['mean_time_in_system_timestep']))

            mean_mts = statistics.mean(
                sum(job.delays) + sum(p for _, p in job.real_routing) for job in self.jobs)  # TODO si può migliorare
            self.mts_stats.append(mean_mts)

    def run(self) -> None:
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

            self.job_manager(job)

    def job_manager(self, job: Job) -> None:
        self.jobs.append(job)
        self.psp.append(job)
