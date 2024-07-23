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
    """
    Calculates the critical value from the Student's t-distribution.
    :param alpha: Significance level (e.g., 0.05 for a two-tailed test with 95% confidence).
    :param n: Number of observations.
    :return: Critical value from the t-distribution.
    """

    return stats.t.ppf(1 - alpha, n - 1)


def analyze_throughput(runs: Sequence[SimSystem], warmup_period: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """
    Analyzes the throughput from multiple simulation runs.
    :param runs: A sequence of simulation system objects containing throughput statistics.
    :param warmup_period: Number of time steps to exclude as warm-up before analyzing throughput.
    :param alpha: Significance level for the confidence interval (default is 0.05 for 95% confidence).
    :return: A tuple containing the mean throughput, variance of throughput, and half-width of the confidence interval.
    """

    n = len(runs)  # Number of simulation runs
    # Extract throughput statistics, ignoring the warm-up period
    sample = [np.mean(run.th_stats[warmup_period:]) for run in runs]

    # Calculate the mean of the throughput sample
    throughput_sample_mean = np.mean(sample)

    # Calculate the variance of the throughput sample
    throughput_sample_variance = statistics.variance(sample, xbar=throughput_sample_mean)

    # Get the critical t-value for the given significance level and number of runs
    t = t_student_critical_value(alpha=alpha, n=n)

    # Calculate the half-width of the confidence interval
    half_interval = t * np.sqrt(throughput_sample_variance / n)

    return throughput_sample_mean, throughput_sample_variance, half_interval


def analyze_wip(runs: Sequence[SimSystem], warmup_period: int, alpha: float = 0.05) -> tuple[list[float], list[float], list[float]]:
    """
    Analyzes the Work-In-Progress (WIP) statistics from multiple simulation runs.
    :param runs: A sequence of simulation system objects containing WIP statistics.
    :param warmup_period: Number of time steps to exclude as warm-up before analyzing WIP.
    :param alpha: Significance level for the confidence interval (default is 0.05 for 95% confidence).
    :return: A tuple containing lists of mean WIP, variance of WIP, and half-widths of confidence intervals for each machine.
    """

    n = len(runs)  # Number of simulation runs

    # Initialize a dictionary to store WIP values for each machine (0 to 5)
    sample = {i: [] for i in range(6)}

    # Iterate on each simulation run
    for run in runs:
        # Extract WIP statistics after the warm-up period and transpose them
        transposed = list(zip(*run.wip_stats[warmup_period:]))  # List of 6 items, one for each machine

        # For each machine, calculate the average WIP values and add them to the corresponding list in the dictionary
        for i, values in enumerate(transposed):
            sample[i].append(np.mean(values))

    # Calculate the mean and variance of the values for each machine
    wip_sample_mean = [np.mean(values) for values in sample.values()]
    wip_sample_variance = [statistics.variance(sample[i], xbar=mean) for i, mean in enumerate(wip_sample_mean)]

    # Calculate the critical t-value for the given significance level and number of runs
    t = t_student_critical_value(alpha=alpha, n=n)

    # Calculate the half-width of the confidence interval for each machine
    half_interval = [t * np.sqrt(std_dev / n) for std_dev in wip_sample_variance]

    return wip_sample_mean, wip_sample_variance, half_interval


def analyze_system_wip(runs: Sequence[SimSystem], warmup_period: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """
    Analyzes the total Work-In-Progress (WIP) statistics for the entire system from multiple simulation runs.
    :param runs: A sequence of simulation system objects containing WIP statistics.
    :param warmup_period: Number of time steps to exclude as warm-up before analyzing WIP.
    :param alpha: Significance level for the confidence interval (default is 0.05 for 95% confidence).
    :return: A tuple containing the mean total WIP, variance of total WIP, and half-width of the confidence interval.
    """

    n = len(runs)  # Number of simulation runs

    total_wip_per_timestep = []

    # Iterate through each simulation run
    for run in runs:
        # Calculate the total WIP for each time step after the warm-up period
        total_wip_per_timestep.append([sum(wip_tuple) for wip_tuple in run.wip_stats[warmup_period:]])

    # Calculate the average total WIP for each run and then the overall average
    sample = [np.mean(tot_wip) for tot_wip in total_wip_per_timestep]
    system_wip_sample_mean = np.mean(sample)

    # Calculate the variance of the total WIP sample
    system_wip_sample_variance = statistics.variance(sample, xbar=system_wip_sample_mean)

    # Calculate the critical t-value for the given significance level and number of runs
    t = t_student_critical_value(alpha=alpha, n=n)

    # Calculate the half-width of the confidence interval for the mean total WIP
    half_interval = t * np.sqrt(system_wip_sample_variance / n)

    return system_wip_sample_mean, system_wip_sample_variance, half_interval


def analyze_mean_delay_in_system(runs: Sequence[SimSystem], warmup_period: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """
    Analyzes the mean delay in the system from multiple simulation runs.
    :param runs: A sequence of simulation system objects containing delay statistics.
    :param warmup_period: Number of time steps to exclude as warm-up before analyzing delays.
    :param alpha: Significance level for the confidence interval (default is 0.05 for 95% confidence).
    :return: A tuple containing the mean delay, variance of the delay, and half-width of the confidence interval.
    """

    n = len(runs)  # Number of simulation runs

    # Collect the mean delay statistics for each simulation run, ignoring the warm-up period
    sample = [np.mean(run.mds_stats[warmup_period:]) for run in runs]

    # Calculate the mean delay across all simulation runs
    delay_sample_mean = np.mean(sample)

    # Calculate the variance of the delay sample
    delay_sample_variance = statistics.variance(sample, xbar=delay_sample_mean)

    # Get the critical t-value for the given significance level and number of runs
    t = t_student_critical_value(alpha=alpha, n=n)

    # Calculate the half-width of the confidence interval for the mean delay
    half_interval = t * np.sqrt(delay_sample_variance / n)

    return delay_sample_mean, delay_sample_variance, half_interval


def analyze_mean_time_in_system(runs: Sequence[SimSystem], warmup_period: int, alpha: float = 0.05) \
        -> tuple[float, float, float]:
    """
    Analyzes the mean time a job spends in the system from multiple simulation runs.
    :param runs: A sequence of simulation system objects containing time-in-system statistics.
    :param warmup_period: Number of time steps to exclude as warm-up before analyzing time in system.
    :param alpha: Significance level for the confidence interval (default is 0.05 for 95% confidence).
    :return: A tuple containing the mean time in system, variance of the time in system, and half-width of the confidence interval.
    """

    n = len(runs)  # Number of simulation runs

    # Collect the mean time-in-system statistics for each simulation run, ignoring the warm-up period
    sample = [np.mean(run.mts_stats[warmup_period:]) for run in runs]

    # Calculate the mean time-in-system across all simulation runs
    time_in_system_sample_mean = np.mean(sample)

    # Calculate the variance of the time-in-system sample
    time_in_system_sample_variance = statistics.variance(sample, xbar=time_in_system_sample_mean)

    # Get the critical t-value for the given significance level and number of runs
    t = t_student_critical_value(alpha=alpha, n=n)

    # Calculate the half-width of the confidence interval for the mean time-in-system
    half_interval = t * np.sqrt(time_in_system_sample_variance / n)

    return time_in_system_sample_mean, time_in_system_sample_variance, half_interval


def analyze_reward(runs: Sequence[SimSystem], warmup_period: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """
    Analyzes the mean reward obtained from multiple simulation runs.
    :param runs: A sequence of simulation system objects containing reward statistics.
    :param warmup_period: Number of time steps to exclude as warm-up before analyzing reward.
    :param alpha: Significance level for the confidence interval (default is 0.05 for 95% confidence).
    :return: A tuple containing the mean reward, variance of the reward, and half-width of the confidence interval.
    """

    n = len(runs)  # Number of simulation runs

    # Collect the mean reward statistics for each simulation run, ignoring the warm-up period
    sample = [np.mean(run.rewards_over_time[warmup_period:]) for run in runs]

    # Calculate the mean reward across all simulation runs
    sample_mean = np.mean(sample)

    # Calculate the variance of the reward sample
    sample_variance = statistics.variance(sample, xbar=sample_mean)

    # Get the critical t-value for the given significance level and number of runs
    t = t_student_critical_value(alpha=alpha, n=n)

    # Calculate the half-width of the confidence interval for the mean reward
    half_interval = t * np.sqrt(sample_variance / n)

    return sample_mean, sample_variance, half_interval


def analyze_action_stat(action_stat: Sequence[Sequence[int]], warmup_period: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """
    Analyzes the action statistics from multiple simulation runs to determine the proportion of actions taken.
    :param action_stat: A sequence of action statistics from multiple episodes, where each episode contains a list of
    actions (0 or 1).
    :param warmup_period: Number of initial time steps to exclude as warm-up before analyzing actions.
    :param alpha: Significance level for the confidence interval (default is 0.05 for 95% confidence).
    :return: A tuple containing the mean percentage of action '1', variance of this percentage, and half-width of the
    confidence interval.
    """

    n = len(action_stat)  # Number of episodes

    # Exclude the warm-up period and process action statistics
    action_stat_welch = [actions_episode[warmup_period:] for actions_episode in action_stat]

    percentage_ones = []

    # Calculate the percentage of '1' actions for each episode
    for episode_actions in action_stat_welch:
        count_zeros = episode_actions.count(0)  # Count of action '0'
        count_ones = episode_actions.count(1)   # Count of action '1'
        if count_zeros != 0:
            percentage = (count_ones / (count_zeros + count_ones))  # Calculate percentage of '1' actions
        else:
            percentage = 1 if count_ones > 0 else 0  # Handle the case with no '0' actions
        percentage_ones.append(percentage)

    # Plot the action statistics for visualization
    plot_action_stat(percentage_ones)

    # Analyze the action statistics
    sample = percentage_ones
    sample_mean_1 = np.mean(sample)
    sample_variance_1 = statistics.variance(sample, xbar=sample_mean_1)
    t = t_student_critical_value(alpha=alpha, n=n)
    half_interval_1 = t * np.sqrt(sample_variance_1 / n)

    return sample_mean_1, sample_variance_1, half_interval_1


def action_stat_print(action_stat: Sequence[Sequence[int]], warmup_period: int) -> None:
    """
    Analyzes and prints statistics related to the action distribution of a series of simulation runs.
    :param action_stat: A sequence of action statistics from multiple episodes, where each episode contains a list of
    actions (0 or 1).
    :param warmup_period: Number of initial time steps to exclude as warm-up before analyzing actions.
    """

    # Retrieve the significance level from the parameters for statistical analysis
    alpha = welch_params['analyze_throughput']['alpha']

    # Analyze the action statistics using the provided warm-up period and significance level
    (action_1_sample_mean, action_1_sample_variance, action_1_half_interval) = analyze_action_stat(
            action_stat,
            warmup_period=warmup_period,
            alpha=alpha)

    # Print the results of the action statistics analysis
    actions_table(action_1_sample_mean, action_1_sample_variance, action_1_half_interval)


def output_analyze(system_collection: list[SimSystem], warmup_period: int) -> None:
    """
    Analyzes and outputs statistics for various performance metrics from a collection of simulation systems.
    :param system_collection: A list of SimSystem instances representing the simulation systems to be analyzed.
    :param warmup_period: Number of initial time steps to exclude as warm-up before analyzing the results.
    """

    # If throughput sampling is enabled in the configuration
    if config['throughput_sampling']:
        # Retrieve significance level for throughput analysis
        alpha = welch_params['analyze_throughput']['alpha']

        # Analyze throughput statistics
        throughput_sample_mean, throughput_sample_variance, half_interval = analyze_throughput(
            system_collection,
            warmup_period=warmup_period,
            alpha=alpha)

        # Output throughput statistics
        throughput_table(throughput_sample_mean, throughput_sample_variance, half_interval)

    # If WIP (Work In Progress) sampling is enabled in the configuration
    if config['wip_sampling']:
        # Retrieve significance level for WIP analysis
        alpha = welch_params['analyze_wip']['alpha']

        # Analyze WIP statistics
        wip_sample_mean, wip_sample_variance, wip_half_interval = (
            analyze_wip(
                system_collection,
                warmup_period=warmup_period,
                alpha=alpha)
        )

        # Output WIP statistics and plot WIP data
        wip_table(wip_sample_mean, wip_sample_variance, wip_half_interval)
        wip_plt(wip_sample_mean, wip_sample_variance, wip_half_interval, alpha)

        # Analyze system-wide WIP statistics
        system_wip_sample_mean, system_wip_sample_variance, system_half_interval = analyze_system_wip(
            system_collection,
            warmup_period=warmup_period,
            alpha=alpha)

        # Output system-wide WIP statistics
        system_wip_table(system_wip_sample_mean, system_wip_sample_variance, system_half_interval)

    # If mean time in system sampling is enabled in the configuration
    if config['mean_time_in_system_sampling']:
        # Retrieve significance level for mean time in system analysis
        alpha = welch_params['analyze_mts']['alpha']

        # Analyze mean time in system statistics
        mts_sample_mean, mts_sample_variance, mts_half_interval = (
            analyze_mean_time_in_system(
                system_collection,
                warmup_period=warmup_period,
                alpha=alpha)
        )

        # Output mean time in system statistics
        mts_table(mts_sample_mean, mts_sample_variance, mts_half_interval)

    # If mean delay in system sampling is enabled in the configuration
    if config['mean_delay_in_system_sampling']:
        # Retrieve significance level for mean delay in system analysis
        alpha = welch_params['analyze_mds']['alpha']

        # Analyze mean delay in system statistics
        mds_sample_mean, mds_sample_variance, mds_half_interval = (
            analyze_mean_delay_in_system(
                system_collection,
                warmup_period=warmup_period,
                alpha=alpha)
        )

        # Output mean delay in system statistics
        mds_table(mds_sample_mean, mds_sample_variance, mds_half_interval)

    # If reward sampling is enabled in the configuration
    if config['reward']['reward_sampling']:
        min_rewards = []
        max_rewards = []
        for system in system_collection:
            # Calculate and store minimum and maximum rewards for each system
            min_reward = min(system.rewards_over_time)
            min_rewards.append(min_reward)
            max_reward = max(system.rewards_over_time)
            max_rewards.append(max_reward)
            print(f"System {system}: Min reward = {min_reward}, Max reward = {max_reward}")

        # Retrieve significance level for reward analysis
        alpha = welch_params['analyze_mds']['alpha']

        # Analyze reward statistics
        sample_mean, sample_variance, half_interval = (
            analyze_reward(
                system_collection,
                warmup_period=warmup_period,
                alpha=alpha)
        )

        # Output reward statistics
        reward_table(sample_mean, sample_variance, half_interval)
