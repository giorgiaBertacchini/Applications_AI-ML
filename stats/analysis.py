from collections.abc import Sequence

from scipy import stats
import numpy as np
import statistics

from simulation.sim_system import SimSystem


def t_student_critical_value(alpha: float, n: int) -> float:
    return stats.t.ppf(1 - alpha, n - 1)


def analyze_throughput(runs: Sequence[SimSystem], warmup_period: int, alpha: float = 0.05) -> tuple[
    float, float, float]:
    n = len(runs)  # TODO non sarebbero - warmup_period ?
    sample = [np.mean(run.th_stats[warmup_period:]) for run in runs]
    throughput_sample_mean = np.mean(sample)
    throughput_sample_variance = statistics.variance(sample, xbar=throughput_sample_mean)
    t = t_student_critical_value(alpha=alpha, n=n)
    half_interval = t * np.sqrt(throughput_sample_variance / n)
    return throughput_sample_mean, throughput_sample_variance, half_interval


def analyze_wip(runs: Sequence[SimSystem], warmup_period: int, alpha: float = 0.05) -> (
        tuple)[list[float], list[float], list[float]]:
    n = len(runs)  # TODO non sarebbero - warmup_period ?

    # Estrai tutte le tuple wip_stats da ogni simulazione
    all_stats = [stat for run in runs for stat in run.wip_stats[warmup_period:]]

    # Trasponi la lista di tuple per separare i valori per indice
    transposed = list(zip(*all_stats))

    wip_sample_mean = [np.mean(values) for values in transposed]
    wip_sample_variance = [statistics.variance(values, xbar=wip_sample_mean) for values in transposed]  # TODO to fix

    t = t_student_critical_value(alpha=alpha, n=n)
    half_interval = [t * np.sqrt(std_dev / n) for std_dev in wip_sample_variance]
    return wip_sample_mean, wip_sample_variance, half_interval


def analyze_mean_delay():
    pass


def analyze_mts(runs: Sequence[SimSystem], warmup_period: int, alpha: float = 0.05) -> tuple[float, float, float]:
    pass
