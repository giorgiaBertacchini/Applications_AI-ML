import matplotlib.pyplot as plt
import numpy as np

from tabulate import tabulate


def throughput_table(throughput_sample_mean, throughput_sample_variance, half_interval):
    # Calculate the confidence interval
    conf_interval_min = throughput_sample_mean - half_interval
    conf_interval_max = throughput_sample_mean + half_interval

    # Calculate the relative error
    if throughput_sample_mean != 0:
        relative_error = 100 * half_interval / throughput_sample_mean
    else:
        relative_error = "Undefined"

    headers = ["-", "Mean", "Variance", "Half Interval", "Confidence Interval", "Relative Error"]
    table = [["Throughput",
              f"{throughput_sample_mean:.2f}",
              f"{throughput_sample_variance:.2f}",
              f"{half_interval:.2f}",
              f"{conf_interval_min:.2f}, {conf_interval_max:.2f}",
              f"{relative_error if relative_error == 'Undefined' else f'{relative_error:.2f}%'}"]]

    # Stampa della tabella
    print(tabulate(table, headers, tablefmt="pretty"))


def wip_table(wip_sample_mean, wip_sample_variance, wip_half_interval):
    # Creazione della tabella
    table = []
    for i, (mean, var, hi) in enumerate(zip(wip_sample_mean, wip_sample_variance, wip_half_interval)):
        if mean != 0:  # Evitare la divisione per zero
            relative_error = 100 * hi / mean
        else:
            relative_error = 'Undefined'
        confidence_interval = (mean - hi, mean + hi)
        table.append([f'Machine {i + 1}', f"{mean:.2f}", f"{var:.2f}", f"{hi:.2f}",
                      f"{confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}",
                      f"{relative_error:.2f}%" if mean != 0 else relative_error])

    # Definizione delle intestazioni della tabella
    headers = ["Machine", "Mean", "Variance", "Half Interval", "Confidence Interval", "Relative Error"]

    # Stampa della tabella
    print(tabulate(table, headers, tablefmt="pretty"))


def wip_plt(wip_sample_mean, wip_sample_variance, wip_half_interval, alpha):
    machines = np.arange(1, len(wip_sample_mean) + 1)

    # Calcola intervalli di confidenza
    conf_interval_lower = np.array(wip_sample_mean) - np.array(wip_half_interval)
    conf_interval_upper = np.array(wip_sample_mean) + np.array(wip_half_interval)

    # Crea figura per le medie dei campioni
    plt.figure(figsize=(10, 6))
    plt.bar(machines, wip_sample_mean, yerr=wip_half_interval, capsize=5, color='skyblue')
    plt.title('WIP Sample Mean with Confidence Intervals')
    plt.xlabel('Machine')
    plt.ylabel('Mean')
    plt.grid(True)
    plt.xticks(machines)
    plt.show()

    # Crea figura per la varianza dei campioni
    plt.figure(figsize=(10, 6))
    plt.bar(machines, wip_sample_variance, color='lightgreen')
    plt.title('WIP Sample Variance')
    plt.xlabel('Machine')
    plt.ylabel('Variance')
    plt.grid(True)
    plt.xticks(machines)
    plt.show()

    # Crea figura per gli intervalli di confidenza
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

    # Calcolo degli errori relativi e creazione del grafico
    relative_errors = 100 * np.array(wip_half_interval) / np.array(wip_sample_mean)
    plt.figure(figsize=(10, 6))
    plt.bar(machines, relative_errors, color='salmon')
    plt.title('WIP Relative Error')
    plt.xlabel('Machine')
    plt.ylabel('Relative Error (%)')
    plt.grid(True)
    plt.xticks(machines)
    plt.show()


def mds_table(delay_sample_mean, delay_sample_variance, half_interval):
    # Calculate the confidence interval
    conf_interval_min = delay_sample_mean - half_interval
    conf_interval_max = delay_sample_mean + half_interval

    # Calculate the relative error
    if delay_sample_mean != 0:
        relative_error = 100 * half_interval / delay_sample_mean
    else:
        relative_error = "Undefined"

    headers = ["", "Mean", "Variance", "Half Interval", "Confidence Interval", "Relative Error"]
    table = [["Mean delay in system",
             f"{delay_sample_mean:.2f}",
             f"{delay_sample_variance:.2f}",
             f"{half_interval:.2f}",
             f"{conf_interval_min:.2f}, {conf_interval_max:.2f}",
              f"{relative_error if relative_error == 'Undefined' else f'{relative_error:.2f}%'}"]]

    print(tabulate(table, headers=headers, tablefmt="pretty"))


def mts_table(time_in_system_sample_mean, time_in_system_sample_variance, half_interval):
    # Calculate the confidence interval
    conf_interval_min = time_in_system_sample_mean - half_interval
    conf_interval_max = time_in_system_sample_mean + half_interval

    # Calculate the relative error
    if time_in_system_sample_mean != 0:
        relative_error = 100 * half_interval / time_in_system_sample_mean
    else:
        relative_error = "Undefined"

    headers = ["-", "Mean", "Variance", "Half Interval", "Confidence Interval", "Relative Error"]
    table = [["Mean time in system",
              f"{time_in_system_sample_mean:.2f}",
              f"{time_in_system_sample_variance:.2f}",
              f"{half_interval:.2f}",
              f"{conf_interval_min:.2f}, {conf_interval_max:.2f}",
              f"{relative_error if relative_error == 'Undefined' else f'{relative_error:.2f}%'}"]]

    print(tabulate(table, headers=headers, tablefmt="pretty"))
