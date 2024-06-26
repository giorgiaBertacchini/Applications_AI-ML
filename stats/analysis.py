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

    # Inizializza un dizionario con chiavi da 0 a 5, ciascuna con una lista vuota come valore
    sample = {i: [] for i in range(6)}

    # Itera su ogni run di simulazione
    for run in runs:
        # Prende i dati della simulazione ignorando il periodo di warmup e li trasposta
        transposed = list(zip(*run.wip_stats[warmup_period:]))  # lista di 6 elementi

        # Per ciascuna variabile (da 0 a 5) aggiunge la media dei valori alla lista corrispondente nel dizionario
        for i, values in enumerate(transposed):
            sample[i].append(np.mean(values))

    # Calcola la media e la varianza dei valori per ogni macchina
    wip_sample_mean = [np.mean(values) for values in sample.values()]
    wip_sample_variance = [statistics.variance(sample[i], xbar=mean) for i, mean in enumerate(wip_sample_mean)]

    t = t_student_critical_value(alpha=alpha, n=n)
    half_interval = [t * np.sqrt(std_dev / n) for std_dev in wip_sample_variance]
    return wip_sample_mean, wip_sample_variance, half_interval


def analyze_mean_delay():
    pass


def analyze_mts(runs: Sequence[SimSystem], warmup_period: int, alpha: float = 0.05) -> tuple[float, float, float]:
    pass
