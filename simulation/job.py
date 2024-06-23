import simpy

from simpy.events import ProcessGenerator

from collections.abc import Sequence

from simulation.machine import Machine
from utils.logger import Logger


class Job:
    def __init__(
            self,
            env: simpy.Environment,
            end_event: simpy.Event,
            family_group: int,
            routing: Sequence[bool],
            processing_times: Sequence[float],
            machines: Sequence[Machine],
            dd: float
    ):
        self.env = env
        self.end_event = end_event
        self.family_group = family_group
        self.processing_times = processing_times
        self.routing = routing
        self.machines = machines
        self.dd = dd

        self.logger = Logger(self.env)
        self.name = f'Job {id(self)}'

        self.logger.log(f'{self} enters in system...')

    def __str__(self):
        return f"{self.name}\033[0m"

    def main(self) -> ProcessGenerator:
        for machine, processing_time in zip(self.machines, self.processing_times):
            with machine.request() as request:

                # Wait for the server to become available
                yield request

                # Wait for the server to process the customer
                yield self.env.process(machine.process_job(processing_time))

        self.logger.log(f'{self} completed!')
        self.end_event.succeed()
