import numpy as np
from matplotlib import pyplot as plt


class Welch:
    def __init__(
            self,
            process: np.ndarray,
            window_size: int,
            tol: float  # tolerance value to find the steady state
    ) -> None:
        self.process = process
        self.window_size = window_size
        self.tol = tol
        self.replications_mean = np.mean(process, axis=0)
        self.averaged_process = self._welch()
        self.diff, self.warmup_period = self._find_steady_state()

    @staticmethod
    def moving_average(arr: np.ndarray, window_size: int) -> np.ndarray:
        """
        Computes the moving average of a 1D array using a specified window size.
        :param arr: The input 1D array for which the moving average is to be computed.
        :param window_size: The size of the moving average window.
        :return: A 1D array containing the moving average of the input array.
        """

        weights = np.ones(window_size) / window_size
        return np.convolve(arr, weights, mode='valid')

    def _welch(self) -> np.ndarray:
        """
        Applies a moving average to smooth the replication means using a window size and computes the averaged process
        for the Welch's test.
        :return: A 1D numpy array containing the smoothed replication means.
        """

        averaged_process = []
        for i in range(1, self.replications_mean.shape[0] - self.window_size):
            if i <= self.window_size:
                # For the initial part, use a window that grows with the index
                averaged_process.append(self.replications_mean[:2 * i - 1].mean())
            else:
                # For later indices, use a fixed-size window centered around the current index
                averaged_process.append(
                    self.replications_mean[i - self.window_size // 2:i + self.window_size // 2].mean())
        return np.array(averaged_process)

    def _find_steady_state(self) -> tuple[np.ndarray, int]:
        """
        Identifies the point at which the process reaches a steady state by analyzing the smoothed data.
        :return: A tuple containing:
             - `diff`: A numpy array of differences between consecutive smoothed values.
            - `index`: An integer representing the index where the steady state is detected, or -1 if not found.
        """

        # Compute the moving average of the averaged process
        arr = self.moving_average(self.averaged_process, self.window_size)

        # Calculate the differences between consecutive smoothed values
        diff = np.diff(arr.flatten())

        # Find the first index where the difference is less than the tolerance
        for i, d in enumerate(diff):
            if d < self.tol:
                return diff, i + self.window_size

        # Return -1 if no steady state is found
        return diff, -1

    def plot(self):
        """
        Plots the averaged process along with a vertical line indicating the warmup period.
        """

        # Plot the averaged process with a label
        plt.plot(self.averaged_process, label='Averaged Process')

        # Add a vertical line to indicate the warmup period
        plt.axvline(self.warmup_period, color='r', linestyle='--', label=f'Warmup period: {self.warmup_period}')

        # Add a legend to the plot
        plt.legend(loc='best')

        # Display the plot
        plt.show()
