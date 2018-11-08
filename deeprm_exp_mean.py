import src.models.REINFORCE as rl
import numpy as np
import tensorflow as tf
import gym
import scipy.signal
import src.envs.Mao as mao
from src.utils import *
from time import sleep

env = train_env = get_env("configs/env.json", 1541)
test_env = get_env("configs/env.json", None)

sess = tf.Session()
n_seqs = 50
iter = 1000
force_stop = 400
ob = env.reset()
state_dim = np.prod(ob.shape)
action_dim = env.action_space.n
discount = 0.9
num_episodes = 1000
lr = 0.0001
seed = 1234
num_job_sets = 100
batch_size = 50


with open("configs/deeprm_env.json", 'r') as f:
    import json
    model_config = json.load(f)

model = rl.model(
    sess, state_dim, action_dim, lr,
    network_widths=model_config['network_widths'])

aspace = np.arange(action_dim)
chosen_actions = np.zeros(shape=(action_dim,))

class Reward(object):

    def __init__(self, factor, gamma):
        # Reward parameters
        self.factor = factor
        self.gamma = gamma

    # Set step rewards to total episode reward
    def total(self, ep_batch, tot_reward):
        for step in ep_batch:
            step[2] = tot_reward * self.factor
        return ep_batch

    # Set step rewards to discounted reward
    def discount(self, ep_batch):
        x = ep_batch[:, 2]

        discounted = scipy.signal.lfilter([1], [1, -self.gamma], x[::-1], axis=0)[::-1]
        discounted *= self.factor

        for i in range(len(discounted)):
            ep_batch[i,2] = discounted[i]

        return ep_batch


def __main__():
    sess.run(tf.initializers.global_variables())
    reward = Reward(1.0, discount)
    baseline = np.zeros(shape=(n_seqs, force_stop), dtype=float)


    for ite in range(iter):
        S, ADV, A = np.empty(shape=(0, state_dim)), np.empty(shape=(0,)), np.empty(shape=(0,))
        for seq_no in range(n_seqs):
            env.reset(seq_no=seq_no)
            s = env._observe()
            s = flatten(s)
            list_s, list_a, list_r, list_y = [], [], [], []

            episode_buffer = np.empty((0, 5), float)
            for _ in range(force_stop):
                a = model.get_action_dist(s)
                possible_actions = get_possible_actions(env)
                p = np.zeros_like(aspace)
                p[possible_actions] = 1
                p[-1] = 1
                a = a * p
                a /= np.sum(a)
                action = np.random.choice(aspace, p=a)
                chosen_actions[action] += 1
                s2, r, done, info = env.step(action)
                s2 = flatten(s2)
                list_s.append(s)
                list_a.append(action)
                list_r.append(r)
                if done:
                    disc_vec = np.array(list_r, dtype=float)
                    for i in range(1, len(disc_vec)):
                        disc_vec[i:] *= discount
                    for i in range(len(disc_vec)):
                        y_i = np.sum(disc_vec[i:]) / (discount ** i)
                        list_y.append(y_i)
                    list_y = np.array(list_y)
                    if i == 0:
                        baseline[seq_no][ep][:len(list_y)] += list_y
                    else:
                        baseline[seq_no][:len(list_y)] = (
                                baseline[seq_no][:len(list_y)] * 0.9 + list_y * 0.1)
                    if ite >= 3:
                        adv = list_y - baseline[seq_no][:len(list_y)]
                        S = np.vstack([S, list_s])
                        ADV = np.hstack([ADV, adv])
                        A = np.hstack([A, list_a])
                    break
            if ite >= 3:
                loss = model.train(S, A, ADV)
                #print(loss)
                slowdown = get_avg_slowdown(info)
                statement = "[episode %d], ep_l2n: %d slowdown %0.2f, loss : %0.2f" % (ite, _, slowdown, loss)
                print(statement)
        sds = []
        rews = []
        for i_episode in range(30):
            test_env.reset(seq_no=i_episode)
            s = test_env._observe()
            s = flatten(s)
            rew = 0.0
            for t in range(force_stop):
                a = model.get_action_dist(s)
                possible_actions = get_possible_actions(test_env)
                p = np.zeros_like(aspace)
                p[possible_actions] = 1
                p[-1] = 1
                a = a * p
                a /= np.sum(a)

                action = np.random.choice(aspace, p=a)
                s, reward, done, info = test_env.step(action)
                rew += reward
                s =  flatten(s)
                if done:
                    break
            slowdown = get_avg_slowdown(info)
            sds.append(slowdown)
            rews.append(rew)
        print("test slowdown mean : %0.2f, avg rew : %0.2f" % (np.mean(sds), np.mean(rew)))



if __name__ == "__main__":
    __main__()
