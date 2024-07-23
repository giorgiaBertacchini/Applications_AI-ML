import matplotlib.pyplot as plt
import numpy as np
from typing import List

from tabulate import tabulate


def throughput_table(throughput_sample_mean: float, throughput_sample_variance: float, half_interval: float) -> None:
    """
    Prints a formatted table summarizing the throughput analysis results.
    This function calculates and displays the mean, variance, half interval, confidence interval, and relative error of
    the throughput based on the given sample statistics.
    :param throughput_sample_mean: The mean throughput value from the sample.
    :param throughput_sample_variance: The variance of the throughput values from the sample.
    :param half_interval: The half-width of the confidence interval for the throughput mean.
    :return: None
    """
    # Calculate the confidence interval
    conf_interval_min = throughput_sample_mean - half_interval
    conf_interval_max = throughput_sample_mean + half_interval

    # Calculate the relative error
    if throughput_sample_mean != 0:
        relative_error = 100 * half_interval / throughput_sample_mean
    else:
        relative_error = "Undefined"

    # Define the table headers
    headers = ["-", "Mean", "Variance", "Half Interval", "Confidence Interval", "Relative Error"]

    # Define the table content
    table = [["Throughput",
              f"{throughput_sample_mean:.2f}",
              f"{throughput_sample_variance:.2f}",
              f"{half_interval:.2f}",
              f"{conf_interval_min:.2f}, {conf_interval_max:.2f}",
              f"{relative_error if relative_error == 'Undefined' else f'{relative_error:.2f}%'}"]]

    print(tabulate(table, headers, tablefmt="pretty"))


def wip_table(wip_sample_mean, wip_sample_variance, wip_half_interval) -> None:
    """
    Prints a formatted table summarizing the Work-In-Progress (WIP) analysis results for each machine.
    This function generates and displays a table with mean, variance, half interval, confidence interval, and relative
    error for WIP statistics across multiple machines.
    :param wip_sample_mean: List of mean WIP values for each machine.
    :param wip_sample_variance: List of variance values for WIP for each machine.
    :param wip_half_interval: List of half-widths of the confidence interval for WIP for each machine.
    :return: None
    """
    table = []
    for i, (mean, var, hi) in enumerate(zip(wip_sample_mean, wip_sample_variance, wip_half_interval)):
        # Calculate the relative error as a percentage if the mean is not zero
        if mean != 0:
            relative_error = 100 * hi / mean
        else:
            relative_error = 'Undefined'

        # Calculate the confidence interval
        confidence_interval = (mean - hi, mean + hi)

        # Append results to the table list
        table.append([f'Machine {i + 1}', f"{mean:.2f}", f"{var:.2f}", f"{hi:.2f}",
                      f"{confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}",
                      f"{relative_error:.2f}%" if mean != 0 else relative_error])

    # Define the table headers
    headers = ["Machine", "Mean", "Variance", "Half Interval", "Confidence Interval", "Relative Error"]

    print(tabulate(table, headers, tablefmt="pretty"))


def system_wip_table(system_wip_sample_mean: float, system_wip_sample_variance: float, system_half_interval: float) -> None:
    """
    Prints a formatted table summarizing the Work-In-Progress (WIP) analysis results for the entire system.
    This function generates and displays a table with mean, variance, half interval, confidence interval, and relative
    error for the system's total WIP.
    :param system_wip_sample_mean: Mean WIP value for the system.
    :param system_wip_sample_variance: Variance of the WIP value for the system.
    :param system_half_interval: Half-width of the confidence interval for the system's WIP.
    :return: None
    """

    # Calculate the confidence interval
    conf_interval_min = system_wip_sample_mean - system_half_interval
    conf_interval_max = system_wip_sample_mean + system_half_interval

    # Calculate the relative error
    if system_wip_sample_mean != 0:
        relative_error = 100 * system_half_interval / system_wip_sample_mean
    else:
        relative_error = "Undefined"

    # Define the table headers
    headers = ["-", "Mean", "Variance", "Half Interval", "Confidence Interval", "Relative Error"]

    # Prepare the table data
    table = [["System wip",
              f"{system_wip_sample_mean:.2f}",
              f"{system_wip_sample_variance:.2f}",
              f"{system_half_interval:.2f}",
              f"{conf_interval_min:.2f}, {conf_interval_max:.2f}",
              f"{relative_error if relative_error == 'Undefined' else f'{relative_error:.2f}%'}"]]

    print(tabulate(table, headers, tablefmt="pretty"))


def wip_plt(wip_sample_mean, wip_sample_variance, wip_half_interval, alpha) -> None:
    """
    Plots various statistics related to Work-In-Progress (WIP) for each machine.
    This function generates four plots:
    1. **WIP Sample Mean with Confidence Intervals**: Displays the mean WIP values with error bars representing the confidence intervals.
    2. **WIP Sample Variance**: Shows the variance of WIP values for each machine.
    3. **WIP Confidence Intervals**: Illustrates the confidence intervals for the mean WIP values.
    4. **WIP Relative Error**: Plots the relative error percentage for each machine.
    :param wip_sample_mean: List of mean WIP values for each machine.
    :param wip_sample_variance: List of variance of WIP values for each machine.
    :param wip_half_interval: List of half intervals for the confidence intervals of the WIP values.
    :param alpha: Significance level used for confidence interval calculation.
    :return: None
    """

    # Define machine indices for x-axis
    machines = np.arange(1, len(wip_sample_mean) + 1)

    # Calculate confidence intervals
    conf_interval_lower = np.array(wip_sample_mean) - np.array(wip_half_interval)
    conf_interval_upper = np.array(wip_sample_mean) + np.array(wip_half_interval)

    # Create figure for sample averages
    plt.figure(figsize=(10, 6))
    plt.bar(machines, wip_sample_mean, yerr=wip_half_interval, capsize=5, color='skyblue')
    plt.title('WIP Sample Mean with Confidence Intervals')
    plt.xlabel('Machine')
    plt.ylabel('Mean')
    plt.grid(True)
    plt.xticks(machines)
    plt.show()

    # Plot sample variance
    plt.figure(figsize=(10, 6))
    plt.bar(machines, wip_sample_variance, color='lightgreen')
    plt.title('WIP Sample Variance')
    plt.xlabel('Machine')
    plt.ylabel('Variance')
    plt.grid(True)
    plt.xticks(machines)
    plt.show()

    # Plot the confidence intervals
    plt.figure(figsize=(10, 6))
    plt.plot(machines, wip_sample_mean, 'o', label='Sample Mean', markersize=8)
    for i in range(len(machines)):
        plt.plot([machines[i], machines[i]], [conf_interval_lower[i], conf_interval_upper[i]], 'r-', linewidth=2)
    plt.title(f'WIP Confidence Intervals (alpha = {alpha})')
    plt.xlabel('Machine')
    plt.ylabel('Value')
    plt.grid(True)
    plt.xticks(machines)
    plt.legend()
    plt.show()

    # Calculation of relative errors and plot
    relative_errors = 100 * np.array(wip_half_interval) / np.array(wip_sample_mean)
    plt.figure(figsize=(10, 6))
    plt.bar(machines, relative_errors, color='salmon')
    plt.title('WIP Relative Error')
    plt.xlabel('Machine')
    plt.ylabel('Relative Error (%)')
    plt.grid(True)
    plt.xticks(machines)
    plt.show()


def mds_table(delay_sample_mean: float, delay_sample_variance: float, half_interval: float) -> None:
    """
    Prints a table summarizing the statistics for the mean delay in the system.
    This function generates and prints a table that includes:
    - Mean delay
    - Variance of delay
    - Half interval of the confidence interval
    - Confidence interval range
    - Relative error of the delay
    :param delay_sample_mean: The mean delay in the system.
    :param delay_sample_variance: The variance of the delay in the system.
    :param half_interval: The half-width of the confidence interval for the delay mean.
    :return: None
    """

    # Calculate the confidence interval
    conf_interval_min = delay_sample_mean - half_interval
    conf_interval_max = delay_sample_mean + half_interval

    # Calculate the relative error
    if delay_sample_mean != 0:
        relative_error = 100 * half_interval / delay_sample_mean
    else:
        relative_error = "Undefined"

    # Define table headers
    headers = ["", "Mean", "Variance", "Half Interval", "Confidence Interval", "Relative Error"]

    # Create table content
    table = [["Mean delay in system",
              f"{delay_sample_mean:.2f}",
              f"{delay_sample_variance:.2f}",
              f"{half_interval:.2f}",
              f"{conf_interval_min:.2f}, {conf_interval_max:.2f}",
              f"{relative_error if relative_error == 'Undefined' else f'{relative_error:.2f}%'}"]]

    print(tabulate(table, headers=headers, tablefmt="pretty"))


def mts_table(time_in_system_sample_mean: float, time_in_system_sample_variance: float, half_interval: float) -> None:
    """
    Prints a table summarizing the statistics for the mean time in the system.
    This function generates and prints a table that includes:
    - Mean time in the system
    - Variance of time in the system
    - Half interval of the confidence interval
    - Confidence interval range
    - Relative error of the mean time in the system
    :param time_in_system_sample_mean: The mean time spent in the system.
    :param time_in_system_sample_variance: The variance of time spent in the system.
    :param half_interval: The half-width of the confidence interval for the mean time.
    :return: None
    """

    # Calculate the confidence interval
    conf_interval_min = time_in_system_sample_mean - half_interval
    conf_interval_max = time_in_system_sample_mean + half_interval

    # Calculate the relative error
    if time_in_system_sample_mean != 0:
        relative_error = 100 * half_interval / time_in_system_sample_mean
    else:
        relative_error = "Undefined"

    # Define table headers
    headers = ["-", "Mean", "Variance", "Half Interval", "Confidence Interval", "Relative Error"]

    # Create table content
    table = [["Mean time in system",
              f"{time_in_system_sample_mean:.2f}",
              f"{time_in_system_sample_variance:.2f}",
              f"{half_interval:.2f}",
              f"{conf_interval_min:.2f}, {conf_interval_max:.2f}",
              f"{relative_error if relative_error == 'Undefined' else f'{relative_error:.2f}%'}"]]

    print(tabulate(table, headers=headers, tablefmt="pretty"))


def reward_table(sample_mean: float, sample_variance: float, half_interval: float) -> None:
    """
    Prints a table summarizing the statistics for rewards.
    This function generates and prints a table that includes:
    - Mean of the rewards
    - Variance of the rewards
    - Half interval of the confidence interval
    - Confidence interval range
    - Relative error of the mean reward
    :param sample_mean: The mean value of the rewards.
    :param sample_variance: The variance of the rewards.
    :param half_interval: The half-width of the confidence interval for the mean reward.
    :return: None
    """

    # Calculate the confidence interval
    conf_interval_min = sample_mean - half_interval
    conf_interval_max = sample_mean + half_interval

    # Calculate the relative error
    if sample_mean != 0:
        relative_error = 100 * half_interval / sample_mean
    else:
        relative_error = "Undefined"

    # Define table headers
    headers = ["-", "Mean", "Variance", "Half Interval", "Confidence Interval", "Relative Error"]

    # Create table content
    table = [["Reward",
              f"{sample_mean:.2f}",
              f"{sample_variance:.2f}",
              f"{half_interval:.2f}",
              f"{conf_interval_min:.2f}, {conf_interval_max:.2f}",
              f"{relative_error if relative_error == 'Undefined' else f'{relative_error:.2f}%'}"]]

    print(tabulate(table, headers, tablefmt="pretty"))


def plot_action_stat(percentage_ones):
    """
    Plots a bar chart showing the percentage of 1's per episode.
    This function generates a bar chart where each bar represents the percentage of 1's observed in each episode.
    The percentages are scaled to be between 0 and 100.
    :param percentage_ones: A list of percentages representing the proportion of 1's in each episode.
    :return: None
    """

    # Create a new figure for the plot
    plt.figure()

    # Plot a bar chart with percentages scaled to 0-100
    plt.bar(range(len(percentage_ones)), [i * 100 for i in percentage_ones])

    # Set the title of the plot
    plt.title('Percentual of 1 per episode')
    plt.xlabel('Episode')
    plt.ylabel('Percentual of 1')

    # Display the plot
    plt.show()


def actions_table(action_1_sample_mean: float, action_1_sample_variance: float, action_1_half_interval: float) -> None:
    """
    Prints a formatted table displaying statistical results for Action 1.
    This function calculates and displays various statistical measures for Action 1, including:
    - Mean
    - Variance
    - Half interval of the confidence interval
    - Confidence interval
    - Relative error
    :param action_1_sample_mean: Mean value of Action 1 statistics.
    :param action_1_sample_variance: Variance of Action 1 statistics.
    :param action_1_half_interval: Half interval of the confidence interval for Action 1.
    :return: None
    """

    # Calculate the confidence interval
    action_1_conf_interval_min = action_1_sample_mean - action_1_half_interval
    action_1_conf_interval_max = action_1_sample_mean + action_1_half_interval

    # Calculate the relative error
    if action_1_sample_mean != 0:
        action_1_relative_error = 100 * action_1_half_interval / action_1_sample_mean
    else:
        action_1_relative_error = "Undefined"

    # Define table headers
    headers = ["-", "Mean", "Variance", "Half Interval", "Confidence Interval", "Relative Error"]

    # Create table data
    table = [["Action 1",
              f"{action_1_sample_mean:.2f}",
              f"{action_1_sample_variance:.2f}",
              f"{action_1_half_interval:.2f}",
              f"{action_1_conf_interval_min:.2f}, {action_1_conf_interval_max:.2f}",
              f"{action_1_relative_error if action_1_relative_error == 'Undefined' else f'{action_1_relative_error:.2f}%'}"]
             ]

    print(tabulate(table, headers, tablefmt="pretty"))


def utilization_plt(utilization_rates: List[List[float]]) -> None:
    """
    Plots the average utilization rates of machines as a bar chart.
    This function takes a list of utilization rates for multiple machines over several time periods or scenarios,
    calculates the average utilization rate for each machine, and plots it as a bar chart.
    :param utilization_rates: A list of lists where each inner list contains utilization rates for a machine
    across different time periods or scenarios.
    :return: None
    """

    # Convert the list of lists into a 2D numpy array
    utilization_rates_array = np.array(utilization_rates)

    # Calculate the mean of each column
    average_utilization_rates = np.mean(utilization_rates_array, axis=0)

    # Convert average utilization rates to percentag
    average_utilization_rates_percent = [value * 100 for value in average_utilization_rates]

    # Define the machine labels based on the number of machines
    machines = range(1, len(average_utilization_rates_percent) + 1)

    # Create a bar chart for the average utilization rates
    plt.bar(machines, average_utilization_rates_percent, color='skyblue')
    plt.title('Machine Utilization Rates')
    plt.xlabel('Machine')
    plt.ylabel('Utilization Rate (%)')
    plt.grid(True)
    plt.xticks(machines)
    plt.show()
