## Production System

This is a simulation of a simple production system. 

The model is implemented in the folder `/simulation`, using the `simpy` library, composed of the system, of the jobs (as customers) and of 6 machines (as servers).

As system benchmark, in the folder `/push_policy.py` 
- the `push_system.py` push the jobs in the servers chain as soon as these enter the system. 
- in `run.py` file first runs some simulations in order to take welch data (warm-up period) and then runs more simulations which it analyzes and gets statistical data.

To create the ml models the `stable_baseline3` library is used. In `/reinforcement_learning` folder there are in particular:
- the `gym_system.py` that implement a `gymnasium` system which relies on the simpy system and defines the observation space and the action space. In particular, it defines the step function and the reset function.
- the `rl_run.py` file runs the episodes where the reinforcement learning model learns at each time step if to push or not the more urgent job.

To change the configuration, see the `/conf` folder: where there are the duration of simulations, the number of simulation, the time of learning, some time steps, the rl model ... and some switches to turn on/off. 
