import gym
import numpy as np
import src.envs.Mao as mao
from time import sleep
from tqdm import tqdm
env = mao.ClusteringEnv(observation_mode='dict')
num_episodes = 100
max_episode_length = 200

class RAND_model(object):
    def __init__(self):
        pass

    def select_job(self, state_as_dict):
        #print(state_as_dict)
        machine = state_as_dict['machine']
        durations = state_as_dict['job_slot']['lengths']
        res_vecs = state_as_dict['job_slot']['resource_vectors']

        sjf_score = 0
        act = len(durations)  # if no action available, hold
        return np.random.randint(0, act)


model = RAND_model()
slowdowns = []
for i_episode in tqdm(range(num_episodes)):
    ep_reward = 0.0
    ep_ave_max_q = 0.0
    #print("episode %d" % i_episode)
    s = env.reset()
    for curr_ep_len in range(max_episode_length):
        #env.render()
        action = a = model.select_job(s)
        #env.render()
        #sleep(0.05)

        s2, reward, done, info = env.step(action)
        ep_reward += reward
        if done:
            break

        s = s2


    slowdown = env._get_avg_slowdown()

    slowdowns.append(slowdown)

print("avg avg slowdown : %0.2f" % np.mean(slowdowns))





#        if done:
#            print("Episode finished after {} timesteps".format(t+1))
#            break
