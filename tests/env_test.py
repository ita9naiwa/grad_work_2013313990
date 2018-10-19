import sys
sys.path.append('../')
sys.path.append('../src')

import test_base

import gym
import numpy as np
import envs.Mao as mao
from time import sleep



def get_entropy(vec):
    entropy = -np.sum(vec * np.log(vec))
    if np.isnan(entropy):
        return 0
    return entropy

env = mao.ClusteringEnv(p_job_arrival=1.0)

for i_episode in range(1):
    observation = env.reset()
    for t in range(1000):

        env.render()

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        observation = np.array(observation)
        observation = np.reshape(observation, newshape=(np.prod(observation.shape),))
        """
        if True:
            print("action", action)
            print("observed", observation)
            print("reward", reward)
            print("done?", done)
        """
        #print(info)
        sleep(1.0)


#        if done:
#            print("Episode finished after {} timesteps".format(t+1))
#            break
