import simpy


class Logger:
    last_timestamp: float

    def __init__(self, env: simpy.Environment) -> None:
        self.env = env
        Logger.last_timestamp = self.env.now

    def log(self, message: str):
        if self.env.now > Logger.last_timestamp:
            print()

        print(f'[{self.env.now}] {message}')
        Logger.last_timestamp = self.env.now
