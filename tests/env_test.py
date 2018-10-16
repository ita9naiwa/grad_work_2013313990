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

def get_traj(agent, env, max_episode_length=1024, render=False):
    env.reset()
    observations = []
    actions = []
    rewards = []
    policy_dists = []
    infos = []

    observation = env.observation()
    for _ in range(max_episode_length):
        policy_dist = agent.p_given_state(observation)
        # whoa, this is nice!
        action = (np.cumsum(policy_dist) > np.random.rand()).argmax()

        observations.append(observation)
        actions.append(action)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        policy_dists.append(policy_dist)
        infos.append(info)

        if done is True:
            break

        if render is True:
            env.render()
    return {
        "observations": observations,
        "rewards": rewards,
        "actions": actions,
        "info": infos,
        "policy_dists": policy_dists}

env = mao.ClusteringEnv(p_job_arrival=1.0)

for i_episode in range(1):
    observation = env.reset()
    for t in range(1000):

        env.render()

        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
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
