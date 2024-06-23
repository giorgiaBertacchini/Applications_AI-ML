import simpy
import yaml

from typing import List  # TODO va bene?
from collections.abc import Sequence, Callable

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
            routing: Callable[[], List[float]],  # TODO valori random tra 0 e 1, sono 6 valori
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

        for i in range(6):
            self.machines.append(Machine(
                env=self.env,
                capacity=1
            ))  # TODO: passare i parametri giusti

        # Statistics: throughput
        self.th_stats: list[int] = [0]
        self.tot_finished_jobs: int = 0  # TODO da incrementare ogni volta che un job termina
        self.last_total_th = 0

        self.env.process(self.run())
        self.env.process(self.throughput_sampler())

    def throughput_sampler(self):
        while True:
            yield self.env.timeout(60)  # TODO: passare il tempo giusto
            delta = self.tot_finished_jobs - self.last_total_th
            self.th_stats.append(delta)
            self.last_total_th = self.tot_finished_jobs

    def run(self):
        while True:

            # Wait for the next customer to arrive
            yield self.env.timeout(self.inter_arrival_time_distribution())

            # Create a new job
            family_group = self.family_index()

            config_routing = config['families']['f{}'.format(family_group)]['routing']
            family_routing = self.routing()
            new_routing = [True if family_routing[i] <= config_routing[i] else False for i in range(6)]

            processing_time_list = []

            for i in range(sum(new_routing)):
                if family_group == 1:
                    processing_time_list.append(self.f1_processing_time_distribution())
                elif family_group == 2:
                    processing_time_list.append(self.f2_processing_time_distribution())
                else:
                    processing_time_list.append(self.f3_processing_time_distribution())

            end_event = self.env.event()
            job = Job(
                env=self.env,
                end_event=end_event,
                family_group=family_group,
                routing=new_routing,
                processing_times=processing_time_list,
                machines=[server for in_routing, server in zip(new_routing, self.machines) if in_routing],
                dd=self.dd()
            )

            self.jobs.append(job)
            self.env.process(job.main())
            self.env.process(self.job_finished(end_event))

    def job_finished(self, end_event):
        yield end_event
        self.tot_finished_jobs += 1
