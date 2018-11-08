import numpy as np
import scipy
import pickle
import src.envs.Mao as Mao

def get_env(config_path, seed=None):
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    if seed is None:
        np.random.seed(config['test_seed'])
    else:
        np.random.seed(seed)
    env = Mao.ClusteringEnv(
        p_job_arrival=config['p_job_arrival'],
        observation_mode='image',
        episode_size=config['episode_size'],
        force_stop=config['ep_force_stop'],
        num_slots=config['num_slots'],
        max_job_length=config['max_job_length'],
        n_resource_slot_capacities=config['n_resource_slot_capacities'])
    return env

def get_possible_actions(env):
    machine = env.machine.repr()
    job_slot = env.job_slot.repr()
    durations = job_slot['lengths']
    resource_vectors = job_slot['resource_vectors']
    enter_time = job_slot['enter_times']
    num_slots  = len(durations)

    time_horizon = machine.shape[0]
    candidates = []

    for j in range(num_slots):
        if durations[j] is None:
            continue

        for i in range(time_horizon):
            if i + durations[j] >= time_horizon:
                break
            r = machine[i:i + durations[j]] - resource_vectors[j]
            if np.all(r >= 0):
                candidates.append(j)
                break

    return candidates




def calc_entropy(p):
    p = p + 0.00001
    ret = -np.sum(np.log(p) * p)
    if np.isinf(ret):
        return 0
    else:
        return ret



def flatten(m):
    state_dim = np.prod(m.shape)
    return np.reshape(m, newshape=(state_dim,))


def get_avg_slowdown(info):
    sds = []
    for job in info.values():
        if job is None:
            continue
        if job.finish_time < 0:
            continue
        sds.append((job.finish_time - job.enter_time) / job.len)
    return np.mean(sds)

def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        out[i] = x[i] + gamma * out[i+1]

    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out

def get_sjf_action(env):
    machine = env.machine.repr()
    job_slot = env.job_slot.repr()
    sjf_score = 0
    durations = job_slot['lengths']
    resource_vectors = job_slot['resource_vectors']
    enter_time = job_slot['enter_times']
    num_slots  = len(durations)

    time_horizon = machine.shape[0]

    ret = num_slots
    candidates = []
    for j in range(num_slots):
        if durations[j] is None:
            continue
        for i in range(time_horizon):
            if i + durations[j] >= time_horizon:
                break
            r = machine[i:i + durations[j]] - resource_vectors[j]
            q = np.all(r >= 0)
            if q:
                candidates.append((durations[j], enter_time[j], j))
    candidates = sorted(candidates)
    if len(candidates) > 0:
        return candidates[0][2]
    else:
        return ret
