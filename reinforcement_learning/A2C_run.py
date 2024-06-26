import random
import simpy
import yaml

#from stable_baselines3 import A2C

from simulation.sim_system import SimSystem
from reinforcement_learning.gym_system import GymSystem

with open('../conf/sim_config.yaml', 'r') as file:
    config = yaml.safe_load(file)


def run_prog(seed: int | None) -> SimSystem:
    random.seed(seed)

    sim_system = SimSystem(
        env=simpy.Environment(),
        inter_arrival_time_distribution=lambda: random.expovariate(
            lambd=float(config['job_arrival_lambda'])),  # 0.65 minute/job
        family_index=lambda: random.choices(
            [1, 2, 3],
            weights=[
                float(config['families']['f1']['weight']),
                float(config['families']['f2']['weight']),
                float(config['families']['f3']['weight'])
            ]
        )[0],
        f1_processing_time_distribution=lambda: random.gammavariate(
            alpha=float(config['families']['f1']['processing']['alpha']),
            beta=float(config['families']['f1']['processing']['beta'])
        ),
        f2_processing_time_distribution=lambda: random.gammavariate(
            alpha=float(config['families']['f2']['processing']['alpha']),
            beta=float(config['families']['f2']['processing']['beta'])
        ),
        f3_processing_time_distribution=lambda: random.gammavariate(
            alpha=float(config['families']['f3']['processing']['alpha']),
            beta=eval(config['families']['f3']['processing']['beta'])  # eval, because is an expression
        ),
        routing=lambda: [random.random() for _ in range(6)],
        dd=lambda: random.uniform(float(config['dd']['min']), float(config['dd']['max']))
    )

    gym_env = GymSystem(sim_system)

    # TODO

    return sim_system

run_prog(*seeds)
