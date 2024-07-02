from collections.abc import Sequence

from scipy import stats
import numpy as np
import statistics
import yaml

from simulation.sim_system import SimSystem
from stats.welch import Welch
from stats.analysis_view import wip_table, wip_plt, mts_table, mds_table, throughput_table

with open('../conf/sim_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

with open('../conf/welch_params.yaml', 'r') as file:
    welch_params = yaml.safe_load(file)


def t_student_critical_value(alpha: float, n: int) -> float:
    return stats.t.ppf(1 - alpha, n - 1)


def analyze_throughput(runs: Sequence[SimSystem], warmup_period: int, alpha: float = 0.05) -> tuple[
    float, float, float]:
    n = len(runs)
    sample = [np.mean(run.th_stats[warmup_period:]) for run in runs]
    throughput_sample_mean = np.mean(sample)
    throughput_sample_variance = statistics.variance(sample, xbar=throughput_sample_mean)
    t = t_student_critical_value(alpha=alpha, n=n)
    half_interval = t * np.sqrt(throughput_sample_variance / n)
    return throughput_sample_mean, throughput_sample_variance, half_interval


def analyze_wip(runs: Sequence[SimSystem], warmup_period: int, alpha: float = 0.05) -> (
        tuple)[list[float], list[float], list[float]]:
    n = len(runs)

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


def analyze_mean_delay_in_system(runs: Sequence[SimSystem], warmup_period: int, alpha: float = 0.05):  # TODO to CHECK
    n = len(runs)
    sample = [np.mean(run.mds_stats[warmup_period:]) for run in runs]
    delay_sample_mean = np.mean(sample)
    delay_sample_variance = statistics.variance(sample, xbar=delay_sample_mean)
    t = t_student_critical_value(alpha=alpha, n=n)
    half_interval = t * np.sqrt(delay_sample_variance / n)
    return delay_sample_mean, delay_sample_variance, half_interval


def analyze_mean_time_in_system(runs: Sequence[SimSystem], warmup_period: int, alpha: float = 0.05) \
        -> tuple[float, float, float]:  # TODO to CHECK
    n = len(runs)
    sample = [np.mean(run.mts_stats[warmup_period:]) for run in runs]
    time_in_system_sample_mean = np.mean(sample)
    time_in_system_sample_variance = statistics.variance(sample, xbar=time_in_system_sample_mean)
    t = t_student_critical_value(alpha=alpha, n=n)
    half_interval = t * np.sqrt(time_in_system_sample_variance / n)
    return time_in_system_sample_mean, time_in_system_sample_variance, half_interval


def output_analyze(system_collection: list[SimSystem]):
    system_runs_arr = np.array([run.th_stats for run in system_collection])
    welch = Welch(system_runs_arr, window_size=welch_params['welch']['window_size'], tol=welch_params['welch']['tol'])
    welch.plot()

    if config['throughput_sampling']:
        alpha = welch_params['analyze_throughput']['alpha']
        throughput_sample_mean, throughput_sample_variance, half_interval = analyze_throughput(
            system_collection,
            warmup_period=welch.warmup_period,
            alpha=alpha)
        throughput_table(throughput_sample_mean, throughput_sample_variance, half_interval)

    if config['wip_sampling']:
        alpha = welch_params['analyze_wip']['alpha']

        wip_sample_mean, wip_sample_variance, wip_half_interval = (
            analyze_wip(
                system_collection,
                warmup_period=welch.warmup_period,
                alpha=alpha)
        )

        wip_table(wip_sample_mean, wip_sample_variance, wip_half_interval)
        wip_plt(wip_sample_mean, wip_sample_variance, wip_half_interval, alpha)

    if config['mean_time_in_system_sampling']:
        alpha = welch_params['analyze_mts']['alpha']

        mts_sample_mean, mts_sample_variance, mts_half_interval = (
            analyze_mean_time_in_system(
                system_collection,
                warmup_period=welch.warmup_period,
                alpha=alpha)
        )

        mts_table(mts_sample_mean, mts_sample_variance, mts_half_interval)

    if config['mean_delay_in_system_sampling']:
        alpha = welch_params['analyze_mds']['alpha']

        mds_sample_mean, mds_sample_variance, mds_half_interval = (
            analyze_mean_delay_in_system(
                system_collection,
                warmup_period=welch.warmup_period,
                alpha=alpha)
        )

        mds_table(mds_sample_mean, mds_sample_variance, mds_half_interval)