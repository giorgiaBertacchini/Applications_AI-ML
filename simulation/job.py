import simpy

from simpy.events import ProcessGenerator

from collections.abc import Sequence, MutableSequence

from simulation.machine import Machine
from utils.logger import Logger


class Job:
    def __init__(
            self,
            env: simpy.Environment,
            family_group: int,
            routing: Sequence[bool],
            processing_times: Sequence[float],
            machines: Sequence[Machine],
            dd: float
    ):
        self.env = env
        self.family_group = family_group
        self.processing_times = processing_times
        self.routing = routing
        self.machines = machines
        self.dd = dd

        self.delays: MutableSequence[float] = []  # TODO: check if it's mutable
        self.real_routing: MutableSequence[tuple[Machine, float]] = []
        self.done: bool = False

        self.arrival_time: float = self.env.now

        self.logger = Logger(self.env)
        self.name = f'Job {id(self)}'

        #self.logger.log(f'{self} enters in system...')

    def __str__(self):
        return f"{self.name}\033[0m"

    @property
    def time_in_system(self) -> float:
        return sum(self.delays) + sum(p for _, p in self.real_routing)

    def main(self) -> ProcessGenerator:
        self.done = False
        for machine, processing_time in zip(self.machines, self.processing_times):

            with machine.request(job=self) as request:
                # Record the time the customer joined the queue
                queue_entry_time = self.env.now

                # Wait for the server to become available
                yield request

                # Record the time the customer left the queue
                queue_exit_time = self.env.now

                # Calculate the time the customer spent waiting in the queue
                self.delays.append(queue_exit_time - queue_entry_time)

                # Wait for the server to process the customer
                yield self.env.process(machine.process_job(processing_time))

                self.real_routing.append((machine, processing_time))

        #self.logger.log(f'{self} completed!')
        self.done = True
