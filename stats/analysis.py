from collections.abc import Sequence

from scipy import stats
import numpy as np
import statistics
import yaml

from simulation.sim_system import SimSystem
from stats.analysis_view import (wip_table, wip_plt, mts_table, mds_table, throughput_table, system_wip_table,
                                 reward_table, plot_action_stat, actions_table)

with open('../conf/sim_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

with open('../conf/welch_params.yaml', 'r') as file:
    welch_params = yaml.safe_load(file)


def t_student_critical_value(alpha: float, n: int) -> float:
    return stats.t.ppf(1 - alpha, n - 1)


def analyze_throughput(runs: Sequence[SimSystem], warmup_period: int, alpha: float = 0.05) -> tuple[float, float, float]:
    n = len(runs)
    sample = [np.mean(run.th_stats[warmup_period:]) for run in runs]
    throughput_sample_mean = np.mean(sample)
    throughput_sample_variance = statistics.variance(sample, xbar=throughput_sample_mean)
    t = t_student_critical_value(alpha=alpha, n=n)
    half_interval = t * np.sqrt(throughput_sample_variance / n)
    return throughput_sample_mean, throughput_sample_variance, half_interval


def analyze_wip(runs: Sequence[SimSystem], warmup_period: int, alpha: float = 0.05) -> tuple[list[float], list[float], list[float]]:
    n = len(runs)

    # Initialize a dictionary with keys from 0 to 5, each with an empty list as the value
    sample = {i: [] for i in range(6)}

    # Iterate on each simulation run
    for run in runs:
        # Takes the simulation data ignoring the warmup period and transposes it
        transposed = list(zip(*run.wip_stats[warmup_period:]))  # list of 6 items

        # For each variable (0 to 5) adds the average of the values to the corresponding list in the dictionary
        for i, values in enumerate(transposed):
            sample[i].append(np.mean(values))

    # Calculate the mean and variance of the values for each machine
    wip_sample_mean = [np.mean(values) for values in sample.values()]
    wip_sample_variance = [statistics.variance(sample[i], xbar=mean) for i, mean in enumerate(wip_sample_mean)]

    t = t_student_critical_value(alpha=alpha, n=n)
    half_interval = [t * np.sqrt(std_dev / n) for std_dev in wip_sample_variance]
    return wip_sample_mean, wip_sample_variance, half_interval


def analyze_system_wip(runs: Sequence[SimSystem], warmup_period: int, alpha: float = 0.05) -> tuple[float, float, float]:
    n = len(runs)

    total_wip_per_timestep = []
    for run in runs:
        total_wip_per_timestep.append([sum(wip_tuple) for wip_tuple in run.wip_stats[warmup_period:]])

    sample = [np.mean(tot_wip) for tot_wip in total_wip_per_timestep]
    system_wip_sample_mean = np.mean(sample)
    system_wip_sample_variance = statistics.variance(sample, xbar=system_wip_sample_mean)
    t = t_student_critical_value(alpha=alpha, n=n)
    half_interval = t * np.sqrt(system_wip_sample_variance / n)
    return system_wip_sample_mean, system_wip_sample_variance, half_interval


def analyze_mean_delay_in_system(runs: Sequence[SimSystem], warmup_period: int, alpha: float = 0.05) -> tuple[float, float, float]:  # TODO to CHECK
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


def analyze_reward(runs: Sequence[SimSystem], warmup_period: int, alpha: float = 0.05) -> tuple[float, float, float]:
    n = len(runs)
    sample = [np.mean(run.rewards_over_time[warmup_period:]) for run in runs]
    sample_mean = np.mean(sample)
    sample_variance = statistics.variance(sample, xbar=sample_mean)
    t = t_student_critical_value(alpha=alpha, n=n)
    half_interval = t * np.sqrt(sample_variance / n)
    return sample_mean, sample_variance, half_interval


def analyze_action_stat(action_stat: Sequence[Sequence[int]], warmup_period: int, alpha: float = 0.05) -> tuple[float, float, float]:
    n = len(action_stat)
    action_stat_welch = [actions_episode[warmup_period:] for actions_episode in action_stat]

    percentage_ones = []

    # For each episode
    for episode_actions in action_stat_welch:
        # Calculate the percentage of 1 with respect to 0 and add the value to the list
        count_zeros = episode_actions.count(0)
        count_ones = episode_actions.count(1)
        if count_zeros != 0:
            percentage = (count_ones / (count_zeros + count_ones))
        else:
            percentage = 1 if count_ones > 0 else 0
        percentage_ones.append(percentage)

    plot_action_stat(percentage_ones)

    sample = percentage_ones
    sample_mean_1 = np.mean(sample)
    sample_variance_1 = statistics.variance(sample, xbar=sample_mean_1)
    t = t_student_critical_value(alpha=alpha, n=n)
    half_interval_1 = t * np.sqrt(sample_variance_1 / n)

    return sample_mean_1, sample_variance_1, half_interval_1


def action_stat_print(action_stat: Sequence[Sequence[int]], warmup_period: int) -> None:
    alpha = welch_params['analyze_throughput']['alpha']
    (action_1_sample_mean, action_1_sample_variance, action_1_half_interval) = analyze_action_stat(
            action_stat,
            warmup_period=warmup_period,
            alpha=alpha)
    actions_table(action_1_sample_mean, action_1_sample_variance, action_1_half_interval)


def output_analyze(system_collection: list[SimSystem], warmup_period: int) -> None:

    if config['throughput_sampling']:
        alpha = welch_params['analyze_throughput']['alpha']
        throughput_sample_mean, throughput_sample_variance, half_interval = analyze_throughput(
            system_collection,
            warmup_period=warmup_period,
            alpha=alpha)
        throughput_table(throughput_sample_mean, throughput_sample_variance, half_interval)

    if config['wip_sampling']:
        alpha = welch_params['analyze_wip']['alpha']

        wip_sample_mean, wip_sample_variance, wip_half_interval = (
            analyze_wip(
                system_collection,
                warmup_period=warmup_period,
                alpha=alpha)
        )

        wip_table(wip_sample_mean, wip_sample_variance, wip_half_interval)
        wip_plt(wip_sample_mean, wip_sample_variance, wip_half_interval, alpha)

        system_wip_sample_mean, system_wip_sample_variance, system_half_interval = analyze_system_wip(
            system_collection,
            warmup_period=warmup_period,
            alpha=alpha)
        system_wip_table(system_wip_sample_mean, system_wip_sample_variance, system_half_interval)

    if config['mean_time_in_system_sampling']:
        alpha = welch_params['analyze_mts']['alpha']

        mts_sample_mean, mts_sample_variance, mts_half_interval = (
            analyze_mean_time_in_system(
                system_collection,
                warmup_period=warmup_period,
                alpha=alpha)
        )

        mts_table(mts_sample_mean, mts_sample_variance, mts_half_interval)

    if config['mean_delay_in_system_sampling']:
        alpha = welch_params['analyze_mds']['alpha']

        mds_sample_mean, mds_sample_variance, mds_half_interval = (
            analyze_mean_delay_in_system(
                system_collection,
                warmup_period=warmup_period,
                alpha=alpha)
        )

        mds_table(mds_sample_mean, mds_sample_variance, mds_half_interval)

    if config['reward']['reward_sampling']:
        min_rewards = []
        max_rewards = []
        for system in system_collection:
            min_reward = min(system.rewards_over_time)
            min_rewards.append(min_reward)
            max_reward = max(system.rewards_over_time)
            max_rewards.append(max_reward)
            print(f"System {system}: Min reward = {min_reward}, Max reward = {max_reward}")

        #print(f"Min reward: {min(min_rewards)}, Max reward: {max(max_rewards)}")

        alpha = welch_params['analyze_mds']['alpha']

        sample_mean, sample_variance, half_interval = (
            analyze_reward(
                system_collection,
                warmup_period=warmup_period,
                alpha=alpha)
        )

        reward_table(sample_mean, sample_variance, half_interval)
