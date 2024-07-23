import simpy
import matplotlib.pyplot as plt

from collections import defaultdict
from collections.abc import Generator

from simpy.resources.resource import Request, Release


class Machine(simpy.Resource):
    def __init__(self, env: simpy.Environment, capacity: int) -> None:
        super().__init__(env, capacity)
        self.env = env
        self.worked_time: float = 0

        self._queue_history: dict[int, float] = defaultdict(float)
        self._qt: list[tuple[float, int]] = []
        self._ut: list[tuple[float, int]] = [(0, 0)]

        self._last_queue_level: int = 0
        self._last_queue_level_timestamp: float = 0

    @property
    def average_queue_length(self) -> float:
        """
        Calculate the average queue length over the simulation time.
        :return: The average queue length.
        """

        return sum(queue_length * time for queue_length, time in self._queue_history.items()) / self.env.now

    @property
    def utilization_rate(self) -> float:
        """
        Calculate the server utilization rate over the simulation time.
        :return: The server utilization rate.
        """

        return self.worked_time / self.env.now

    @property
    def idle_time(self) -> float:
        """
        Calculate the total time the server has been idle.
        :return: The total idle time.
        """

        return self.env.now - self.worked_time

    @property
    def queue_length(self) -> int:
        """
        Returns the current length of the queue.
        :returns: The number of items in the queue.
        """

        return len(self.queue)

    def _update_ut(self) -> None:
        """
        Update the server utilization over time.
        :return: None
        """

        # status = 1 if the server is busy or there are customers in the queue, 0 otherwise
        status = int(self.count == 1 or len(self.queue) > 0)

        # If the server utilization has not changed, do not update the list
        if self._ut and self._ut[-1][1] == status:
            return

        self._ut.append((self.env.now, status))

    def _update_qt(self) -> None:
        """
        Update the queue length over time.
        :return: None
        """

        self._qt.append((self.env.now, len(self.queue)))

    def _update_queue_history(self, _) -> None:
        """
        Update the queue history with the current queue length and time.
        To be used when a request is instantiated, and as a callback function to the request as well.
        :param _: Ignored parameter.
        :return: None
        """

        # Update the time spent at the last queue level
        self._queue_history[self._last_queue_level] += self.env.now - self._last_queue_level_timestamp

        # Update the last queue level and timestamp
        self._last_queue_level_timestamp = self.env.now
        self._last_queue_level = len(self.queue)

        # Update the queue length over time
        self._update_qt()

    def request(self, job: object) -> Request:
        """
        Request the server.
        Overrides the `request` method of `simpy.Resource`.
        Updates the queue history accordingly.
        :return: The request object.
        """

        # Request the server
        request = super().request()

        # Monkey patch the request object to include a reference to the customer
        request.job = job

        # Update the queue history
        self._update_queue_history(None)

        # Add a callback to update the queue history when the request is triggered
        request.callbacks.append(self._update_queue_history)

        return request

    def release(self, request: Request) -> Release:
        """
        Release the server.
        Overrides the `release` method of `simpy.Resource`.
        Updates the server utilization accordingly.
        :param request: The request object.
        :return: The release object.
        """

        release = super().release(request)
        self._update_ut()
        return release

    def process_job(self, processing_time: float) -> Generator[simpy.Event, None, None]:
        """
        Simulates the processing of a job by the server.
        Updates the instance variable 'worked_time' with the processing time.
        :param processing_time: The time required to process the job in seconds.
        :returns: A SimPy event representing the delay for the processing time.
        """

        # Simulate the processing time
        yield self.env.timeout(processing_time)

        # Update the total time the server has been busy
        self.worked_time += processing_time

    def plot_qt(self) -> None:
        """
        Plot the queue length over time.
        :return: None
        """

        x, y = zip(*self._qt)
        plt.step(x, y, where='pre')
        plt.fill_between(x, y, step='pre', alpha=1.0)
        plt.title('Q(t): Queue length over time')
        plt.xlabel('Simulation Time')
        plt.ylabel('Queue Length')
        plt.show()

    def plot_ut(self) -> None:
        """
        Plot the server utilization over time.
        :return: None
        """

        ut = self._ut + [(self.env.now, self._ut[-1][1])]
        x, y = zip(*ut)
        plt.step(x, y, where='post')
        plt.fill_between(x, y, step='post', alpha=1.0)
        plt.title('U(t): Server utilization over time')
        plt.xlabel('Simulation Time')
        plt.ylabel('Utilization rate')
        plt.show()
