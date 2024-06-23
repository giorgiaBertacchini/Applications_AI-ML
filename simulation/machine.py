import simpy

from collections.abc import Generator


class Machine(simpy.Resource):
    def __init__(self, env: simpy.Environment, capacity: int) -> None:
        super().__init__(env, capacity)
        self.env = env

    def process_job(self, processing_time: float) -> Generator[simpy.Event, None, None]:
        yield self.env.timeout(processing_time)
