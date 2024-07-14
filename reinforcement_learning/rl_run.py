import random
from typing import List

import simpy
import yaml
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from stable_baselines3 import A2C, DQN, PPO

from simulation.sim_system import SimSystem
from reinforcement_learning.gym_system import GymSystem
from reinforcement_learning.plotting import plt_reward_over_episodes

from stats.analysis import output_analyze, action_stat_print
from stats.analysis_view import utilization_plt
from stats.welch import Welch

from datetime import datetime

seeds = [
    6145, 4098, 6148, 2055, 4103, 8200, 2058, 2062, 6159, 6161, 18, 2067, 20, 4117, 2039, 4115, 6164, 24, 4123, 4124, 6173, 29, 8221, 2081, 6177, 40, 4137, 42, 45, 6189, 8240, 2098, 4146, 8249, 8252, 8253, 4159, 8259, 4164, 6213, 4168, 72, 6220, 2124, 8270, 8268, 80, 4178, 6229, 8278, 87, 4185, 2139, 92, 8283, 8286, 4192, 8290, 6245, 4198, 104, 2153, 4207, 6257, 2161, 113, 4210, 4213, 4214, 8311, 120, 6265, 4219, 126, 2175, 6274, 4226, 6275, 4232, 2186, 138, 2189, 4240, 145, 8338, 2196, 149, 8342, 8343, 2200, 6297, 154, 155, 156, 8350, 2209, 8356, 2213, 8358, 165, 2216, 2218, 6317, 6318, 6320, 177, 4274, 8371, 8372, 2230, 6327, 184, 186, 4284, 2239, 2242, 2243, 6340, 6346, 4305, 6353, 8411, 220, 8413, 8414, 6368, 6370, 6374, 4327, 4329, 6378, 2284, 237, 4336, 6384, 244, 2295, 6392, 2298, 8444, 252, 6396, 2304, 4353, 259, 6406, 8456, 2313, 4363, 8460, 270, 6416, 273, 2322, 2323, 6417, 6426, 6430, 8480, 2338, 293, 2342, 8487, 4392, 8491, 300, 4396, 4398, 2352, 8496, 2356, 8505, 4413, 8513, 8515, 4420, 325, 4421, 323, 8520, 6473, 8522, 2380, 8525, 6478, 336, 4433, 338, 337, 8535, 4442, 2395, 4450, 6502, 2406, 2409, 2410, 4460, 8558, 2415, 8563, 373, 4470, 2423, 6518, 8571, 4479, 383, 8582, 391, 390, 2441, 399, 6544, 401, 8598, 2458, 4508, 2460, 413, 2464, 6561, 2465, 8608, 422, 423, 8615, 6566, 6567, 6571, 6575, 8624, 440, 2492, 8637, 446, 6594, 4546, 6595, 4547, 454, 8650, 463, 464, 8657, 8655, 2516, 2518, 6617, 6619, 4572, 477, 4579, 6631, 8679, 8685, 6638, 2543, 6643, 8691, 6644, 6648, 506, 4603, 2563, 2565, 8714, 525, 8718, 4626, 4628, 2582, 2583, 6684, 6689, 6690, 6691, 8740, 8741, 550, 2606, 2607, 6704, 4658, 4666, 8763, 8764, 574, 4671, 578, 584, 8776, 2633, 2635, 8784, 8794, 2655, 2667, 6765, 624, 625, 6770, 2672, 2676, 6775, 637, 6782, 2685, 4734, 4738, 8835, 8834, 2696, 6799, 8851, 6804, 2709, 6806, 8854, 4762, 2718, 8863, 673, 8869, 677, 6824, 681, 6828, 4788, 8885, 693, 8891, 702, 705, 8897, 2757, 6853, 4809, 2762, 715, 2763, 2768, 8919, 4823, 8921, 4826, 6881, 2788, 6884, 4838, 4837, 746, 4843, 8945, 8948, 4853, 2806, 759, 2811, 764, 6909, 766, 8964, 2825, 8973, 781, 8975, 2832, 6929, 6931, 788, 8981, 8982, 6935, 791, 2839, 4890, 6939, 6941, 8990, 8993, 801, 2849, 2853, 2854, 6950, 2861, 4912, 2865, 818, 9010, 6965, 6966, 4921, 4924, 4925, 6977, 4932, 2886, 9031, 9035, 6988, 845, 844, 2895, 6995, 2906, 2907, 4955, 2908, 7007, 9055, 4963, 2916, 9064, 2924, 2931, 884, 9077, 9075, 9079, 7031, 9081, 7029, 9083, 9084, 9085, 2934, 2944, 7043, 2948, 4997, 4999, 9095, 5003, 7054, 9103, 913, 7057, 7058, 2964, 918, 9112, 921, 2969, 2972, 7069, 2975, 5025, 9130, 5038, 9134, 9138, 2995, 949, 953, 9146, 9149, 957, 960, 9153, 7108, 9157, 968, 5065, 5069, 3025, 7121, 978, 3028, 9169, 7122, 9172, 992, 3045, 7147, 5100, 1003, 9198, 9202, 3058, 3060, 1010, 1014, 9206, 9208, 7159, 3066, 3064, 1016, 9213, 1022, 3062, 7163, 5121, 9224, 1033, 1034, 5131, 7181, 1038, 7184, 3088, 5136, 5138, 1048, 5146, 1054, 1056, 7200, 7204, 9253, 3109, 3117, 7214, 5167, 1072, 1071, 5170, 7220, 7221, 7225, 7226, 1086, 9283, 9285, 1094, 9286, 1095, 7241, 3143, 9291, 1099, 9293, 3142, 7247, 9295, 5201, 9298, 3149, 3156, 9301, 5206, 9303, 7259, 5212, 3167, 7264, 9315, 5220, 5219, 1126, 1128, 9320, 1130, 7276, 7277, 7279, 9328, 9333, 1147, 9343, 3201, 7298, 9347, 7299, 1158, 5255, 3209, 1161, 5261, 7311, 5265, 1178, 5275, 1181, 9373, 5278, 3233, 7333, 3238, 5287, 3240, 3241, 3242, 5289, 5288, 7341, 1198, 3239, 7349, 9400, 5305, 5310, 5311, 1214, 5316, 5318, 9419, 7373, 9422, 9424, 7377, 3280, 7379, 9429, 5334, 3287, 3286, 7384, 3295, 7392, 1248, 1251, 5348, 9447, 9448, 1257, 3305, 7408, 9456, 7413, 3322, 5372, 7421, 3327, 9473, 9477, 7436, 9489, 3346, 7445, 5402, 7452, 9505, 7461, 3366, 1319, 5418, 9515, 7475, 3380, 9525, 1334, 7479, 9528, 5433, 5430, 9532, 7484, 5445, 5446, 7495, 9544, 7496, 9541, 7493, 3398, 3405, 7501, 9550, 9549, 5448, 7500, 1365, 7514, 1373, 3422, 9565, 3424, 7521, 3426, 9571, 3427, 1378, 3431, 5480, 3432, 1387, 3435, 7532, 5488, 7536, 3442, 1397, 1400, 9593, 7546, 1414, 9607, 3464, 9614, 1424, 1097, 5522, 9618, 7574, 1431, 1432, 9624, 5532, 3485, 9633, 9634, 1443, 1444, 7590, 9643, 1459, 3509, 3515, 3517, 5567, 7617, 3522, 3521, 7622, 9672, 5577, 5579, 1483, 5581, 3534, 7631, 5586, 1493, 3146, 1496, 3545, 5594, 9691, 1502, 5598, 9698, 1507, 5604, 3557, 9701, 1511, 9704, 5608, 7653, 5614, 9711, 1523, 9715, 9720, 3577, 3579, 3580, 5629, 3582, 7677, 5634, 3587, 5636, 3592, 3596, 9743, 3600, 1553, 7698, 7703, 1561, 3622, 7719, 7718, 3625, 3626, 9771, 5676, 9770, 7726, 1578, 1585, 5686, 5687, 9784, 3644, 1598, 3647, 9791, 5697, 9794, 7747, 5700, 5701, 9799, 3655, 7755, 3666, 9810, 1623, 7769, 7771, 9819, 5725, 5728, 7779, 1638, 3687, 3688, 1641, 5738, 1646, 9838, 1649, 9843, 7800, 7802, 5759, 3712, 1663, 5764, 7812, 7818, 9869, 3726, 9870, 1679, 7823, 9880, 5785, 5788, 5789, 5790, 3742, 1696, 9889, 5794, 1697, 9885, 3751, 1704, 3753, 7848, 5807, 1712, 7857, 3761, 5812, 7861, 3766, 3767, 7865, 1721, 3771, 9915, 7869, 1726, 3775, 8076, 3777, 7874, 9922, 7876, 3780, 5829, 3783, 5827, 3786, 5834, 9933, 9934, 9941, 5847, 7899, 5852, 1763, 7907, 7911, 5863, 5865, 7913, 1770, 7916, 5871, 5873, 3826, 1777, 1783, 5880, 5882, 9979, 9980, 5885, 3838, 3839, 1792, 9986, 3842, 5892, 3851, 9996, 7952, 3857, 5904, 1808, 7957, 7960, 7965, 7967, 1829, 3879, 1833, 7981, 5936, 7985, 1844, 1854, 3903, 8001, 3913, 1866, 8014, 3921, 3922, 8021, 5974, 3926, 8026, 5979, 3934, 5983, 8032, 5985, 1890, 5987, 1892, 1893, 5993, 1897, 1900, 8045, 3953, 8050, 1908, 1913, 6010, 3970, 1922, 3972, 3974, 3979, 1932, 3980, 8078, 6030, 6032, 3985, 6034, 8080, 8081, 3989, 3991, 3994, 3995, 1956, 8102, 6056, 6060, 1965, 4016, 4017, 8117, 4022, 1974, 4024, 8120, 8122, 6075, 6072, 6077, 6082, 4041, 1993, 1994, 8143, 6095, 8147, 6105, 2010, 4058, 2012, 6106, 6107, 2013, 8160, 6113, 2018, 4066, 6114, 8166, 2024, 4081, 4085, 8182, 8183, 8189, 6143
]

with open('../conf/sim_config.yaml', 'r') as file:
    config = yaml.safe_load(file)


with open('../conf/rl_parameters.yaml', 'r') as file:
    rl_params = yaml.safe_load(file)

with open('../conf/welch_params.yaml', 'r') as file:
    welch_params = yaml.safe_load(file)


def log_scalar(summary_writer, tag, value, step):
    with summary_writer.as_default():
        tf.summary.scalar(tag, value, step=step)


def compute_returns(rewards, gamma):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R  # Calculate discounted return
        returns.insert(0, R)  # Insert at the beginning of the list
    return returns


def new_simpy_env() -> SimSystem:
    return SimSystem(
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


def run_prog(*seeds: int):
    random.seed(seeds[0])

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
            beta=eval(config['families']['f3']['processing']['beta'])
        ),
        routing=lambda: [random.random() for _ in range(6)],
        dd=lambda: random.uniform(float(config['dd']['min']), float(config['dd']['max']))
    )

    gym_env = GymSystem(sim_system)

    model = None
    if rl_params['model'] == 'A2C':
        model = A2C("MultiInputPolicy", gym_env, verbose=1, tensorboard_log=log_dir + "/gym")
        model.learn(total_timesteps=int(rl_params['learning_total_timesteps']))
    elif rl_params['model'] == 'DQN':
        model = DQN("MultiInputPolicy", gym_env, verbose=1, tensorboard_log=log_dir + "/gym")
        model.learn(total_timesteps=int(rl_params['learning_total_timesteps']))
    elif rl_params['model'] == 'PPO':
        model = PPO("MultiInputPolicy", gym_env, verbose=1, tensorboard_log=log_dir + "/gym")
        model.learn(total_timesteps=int(rl_params['learning_total_timesteps']))

    total_reward_over_episodes: list[float] = []
    actions_distribution = []
    sim_system_collection = []
    last_job_count = []
    last_finished_job_count = []
    utilization_rates = []

    # TODO per normalizzare:
    max_wip = float('-inf')

    max_first_job_processing_times = float('-inf')

    min_slack = float('inf')
    max_slack = float('-inf')

    min_reward = float('inf')
    max_reward = float('-inf')

    for e in range(int(rl_params['episode_count'])):
        gym_env.simpy_env_reset(new_simpy_env())
        obs, _ = gym_env.reset(seed=seeds[e])
        total_reward = 0
        reward_over_episode: list[float] = []

        for i in range(int(rl_params['time_step_count'])):

            #print(f"Count Jobs: {len(gym_env.simpy_env.jobs)}")
            #print(f"Count finished Jobs: {gym_env.simpy_env.finished_jobs}")
            action, _states = model.predict(obs)
            next_state, reward, done, truncated, info = gym_env.step(action)
            gym_env.render("human")

            #print(f"Reward: {reward}")
            #print(f"gym_env.action_stat: {gym_env.action_stat.count(0)}, {gym_env.action_stat.count(1)}")
            #print(f"Not job to push: {gym_env.not_job_to_push_action_0}, {gym_env.not_job_to_push_action_1}")
            log_scalar(summary_writer, f"Reward {seeds[e]}", reward, e)
            reward_over_episode.append(reward)
            total_reward += reward

            if done or truncated:
                print(f"Episodio terminato! Reward: {reward_over_episode}; i: {i}")
                break

            if 'wip' in next_state and max(next_state['wip']) > max_wip:
                max_wip = max(next_state['wip'])

            if 'first_job_processing_times' in next_state and max(
                    next_state['first_job_processing_times']) > max_first_job_processing_times:
                max_first_job_processing_times = max(next_state['first_job_processing_times'])

            if 'slack' in next_state and next_state['slack'] > max_slack:
                max_slack = next_state['slack']
            if 'slack' in next_state and next_state['slack'] < min_slack:
                min_slack = next_state['slack']

            if reward < min_reward:
                min_reward = reward
            if reward > max_reward:
                max_reward = reward

            obs = next_state

            log_scalar(summary_writer, f"Reward in episode {seeds[e]}", reward, i)

        last_job_count.append(len(gym_env.simpy_env.jobs))
        last_finished_job_count.append(gym_env.simpy_env.finished_jobs)
        actions_distribution.append(gym_env.action_stat)

        utilization_rates.append([machine.utilization_rate for machine in
                                  gym_env.simpy_env.machines])  # Append the utilization rates of the machines

        sim_system_collection.append(gym_env.simpy_env)

        #print(f"gym_env.action_stat: {gym_env.action_stat.count(0)}, {gym_env.action_stat.count(1)}")
        #print(f"Not job to push: {gym_env.not_job_to_push_action_0}, {gym_env.not_job_to_push_action_1}")

        log_scalar(summary_writer, f"Job Count {seeds[e]}", len(gym_env.simpy_env.jobs), e)
        log_scalar(summary_writer, f"Finished Job Count {seeds[e]}", gym_env.simpy_env.finished_jobs, e)

        log_scalar(summary_writer, f"gym_env.action_stat {seeds[e]}", gym_env.action_stat.count(1), e)
        log_scalar(summary_writer, f"Not job to push - 0 {seeds[e]}", gym_env.not_job_to_push_action_0, e)
        log_scalar(summary_writer, f"Not job to push - 1 {seeds[e]}", gym_env.not_job_to_push_action_1, e)
        log_scalar(summary_writer, f"Correctly delivered {seeds[e]}", gym_env.simpy_env.correctly_finished_jobs, e)
        log_scalar(summary_writer, f"Tot delivered {seeds[e]}", gym_env.simpy_env.finished_jobs, e)

    if not rl_params['normalize_state']:
        log_scalar(summary_writer, f"Max WIP", max_wip, 0)
        log_scalar(summary_writer, f"Max First Job Processing Times", max_first_job_processing_times, 0)
        log_scalar(summary_writer, f"Max Slack", float(max_slack), 0)
        log_scalar(summary_writer, f"Min Slack", float(min_slack), 0)

    if not rl_params['normalize_reward']:
        log_scalar(summary_writer, f"Min Reward", float(min_reward), 0)
        log_scalar(summary_writer, f"Max Reward", float(max_reward), 0)

    #print(f"Last Job Count: {last_job_count}")
    #print(f"Last Finished Job Count: {last_finished_job_count}")

    #print(f"Max WIP: {max_wip}")
    #print(f"Max First Job Processing Times: {max_first_job_processing_times}")
    #print(f"Max Slack: {max_slack}")
    #print(f"Min Slack: {min_slack}")
    #print(f"Min Reward: {min_reward}")
    #print(f"Max Reward: {max_reward}")

    gym_env.close()

    # TODO to test
    utilization_plt(utilization_rates)
    return actions_distribution, total_reward_over_episodes, sim_system_collection

log_dir = f"./logs/test/{rl_params['model']}/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
summary_writer = tf.summary.create_file_writer(log_dir)

welch_simulations_number = int(welch_params['welch']['simulations_number'])
stat_simulations_number = int(welch_params['rl_system']['stat_simulations_number'])

num_seeds = len(seeds)
_, _, sim_system_collection = run_prog(*seeds[:welch_simulations_number])
system_runs_arr = np.array([run.th_stats for run in sim_system_collection])
plt.plot(system_runs_arr.T)
plt.show()

welch = Welch(system_runs_arr, window_size=welch_params['welch']['window_size'], tol=welch_params['welch']['tol'])
welch.plot()

seed_count = welch_simulations_number + int(rl_params['episode_count'])
actions_distribution, total_reward_over_episodes, sim_system_collection = run_prog(*seeds[welch_simulations_number:seed_count])

title = ''
if rl_params['model'] == 'A2C':
    title = 'ActorCritic (A2C)'
elif rl_params['model'] == 'DQN':
    title = 'Deep Q-Learning (DQN)'
elif rl_params['model'] == 'PPO':
    title = 'Proximal Policy Optimization (PPO)'
plt_reward_over_episodes(total_reward_over_episodes, title)  # TODo posso togliere
output_analyze(sim_system_collection, welch.warmup_period)
action_stat_print(actions_distribution, welch.warmup_period)


# tensorboard --logdir=./reinforcement_learning/logs/test/