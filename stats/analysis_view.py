import matplotlib.pyplot as plt
import numpy as np

from tabulate import tabulate


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
    plt.title(f'Confidence Intervals (alpha = {alpha})')
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
    plt.title('Relative Error')
    plt.xlabel('Machine')
    plt.ylabel('Relative Error (%)')
    plt.grid(True)
    plt.xticks(machines)
    plt.show()