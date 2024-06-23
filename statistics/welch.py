from collections.abc import Sequence

from scipy import stats

import numpy as np
import statistics
from matplotlib import pyplot as plt

from simulation.sim_system import SimSystem


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
        weights = np.ones(window_size) / window_size
        return np.convolve(arr, weights, mode='valid')

    def _welch(self) -> np.ndarray:
        averaged_process = []
        for i in range(1, self.replications_mean.shape[0] - self.window_size):
            if i <= self.window_size:
                averaged_process.append(self.replications_mean[:2 * i - 1].mean())
            else:
                averaged_process.append(
                    self.replications_mean[i - self.window_size // 2:i + self.window_size // 2].mean())
        return np.array(averaged_process)

    def _find_steady_state(self) -> tuple[np.ndarray, int]:
        arr = self.moving_average(self.averaged_process, self.window_size)
        diff = np.diff(arr.flatten())
        for i, d in enumerate(diff):
            if d < self.tol:
                return diff, i + self.window_size
        return diff, -1

    def plot(self):
        plt.plot(self.averaged_process, label='Averaged Process')
        plt.axvline(self.warmup_period, color='r', linestyle='--', label=f'Warmup period: {self.warmup_period}')
        plt.legend(loc='best')
        plt.show()


def t_student_critical_value(alpha: float, n: int) -> float:
    return stats.t.ppf(1 - alpha, n - 1)


def analyze_throughput(runs: Sequence[SimSystem], warmup_period: int, alpha: float = 0.05) -> tuple[float, float, float]:
    n = len(runs)
    sample = [statistics.mean(run.th_stats[warmup_period:]) for run in runs]
    throughput_sample_mean = statistics.mean(sample)
    throughput_sample_variance = statistics.variance(sample, xbar=throughput_sample_mean)
    t = t_student_critical_value(alpha=alpha, n=n)
    half_interval = t * np.sqrt(throughput_sample_variance / n)
    return throughput_sample_mean, throughput_sample_variance, half_interval
