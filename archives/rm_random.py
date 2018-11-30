import gym
import numpy as np
import src.envs.Mao as mao
from time import sleep
from src.utils import get_avg_slowdown

def get_rand_action(observation):
    machine = observation['machine']
    job_slot = observation['job_slot']
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
                candidates.append(j)

    if len(candidates) > 0:
        return np.random.choice(candidates)
    else:
        return ret


def get_entropy(vec):
    entropy = -np.sum(vec * np.log(vec))
    if np.isnan(entropy):
        return 0
    return entropy

max_ep_size = 100
env = mao.ClusteringEnv(p_job_arrival=0.7, observation_mode='def',
        episode_size=50, force_stop=100, num_slots=5,
        n_resource_slot_capacities=(20, 20))

np.random.seed(1)

sds = []
for i_episode in range(10):
    observation = env.reset()
    for t in range(max_ep_size):
        #env.render()
        action = get_rand_action(observation)
        observation, reward, done, info = env.step(action)
        #print(info)
        #sleep(0.1)
        if done:
            break
    slowdown = get_avg_slowdown(info)
    sds.append(slowdown)

print(np.mean(sds))

#        if done:
#            print("Episode finished after {} timesteps".format(t+1))
#            break
