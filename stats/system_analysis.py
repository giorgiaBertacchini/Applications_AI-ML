from simulation.sim_system import SimSystem
import statistics
import matplotlib.pyplot as plt


def plot_inter_arrival_time(system: SimSystem):  # TODO da plottare
    average_customer_inter_arrival_time = statistics.mean(system.jobs_inter_arrival_times)
    print(f"Average Customer Inter-arrival time = {average_customer_inter_arrival_time:.2f} minutes")
    plt.hist(system.jobs_inter_arrival_times, bins=50)
    plt.title('Inter-arrival time distribution')
    plt.show()
