from tqdm import tqdm
import src.models.ddpg as ddpg
from src.models.buffer import ReplayBuffer
from src.noise import OrnsteinUhlenbeckActionNoise
from src.summary import summary
import numpy as np
import tensorflow as tf
import gym
import scipy.signal
import src.envs.Mao as mao
from src.utils import *

np.random.seed(1)
force_stop = 100
num_slots = 10
force_stop = 400
env = get_env("configs/env.json", 1541)
test_env = get_env("configs/env.json", None)
sess = tf.Session()
ob = env.reset()
state_dim = np.prod(ob.shape)
action_dim = env.action_space.n
discount = 0.9
batch_size = 4
num_episodes = 1000
render = True
buffer_size = 3000
actor_lr = 1e-5
critic_lr = 1e-4
tau = 0.001
seed = 1234
model = ddpg.DDPG(sess, action_dim, state_dim, actor_lr, critic_lr, tau=tau, use_softmax=False)

def get_impossible_actions(observation):
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
            candidates.append(j)
            continue
        for i in range(time_horizon):
            if i + durations[j] >= time_horizon:
                break
            r = machine[i:i + durations[j]] - resource_vectors[j]
            q = np.all(r >= 0)
            if q:
                #candidates.append(j)
                pass
            else:
                candidates.append(j)

    return candidates

def flatten(m, a=state_dim):
    return np.reshape(m, newshape=(state_dim,))

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
        x = ep_batch[:,2]

        discounted = scipy.signal.lfilter([1], [1, -self.gamma], x[::-1], axis=0)[::-1]
        discounted *= self.factor

        for i in range(len(discounted)):
            ep_batch[i,2] = discounted[i]

        return ep_batch



def __main__():
    FILTER_OUTPUT = False
    sess.run(tf.initializers.global_variables())
    for ite in range(1000):
        replay_buffer = ReplayBuffer(buffer_size)
        seq_list = np.arange(50)
        np.random.shuffle(seq_list)
        seq_list = seq_list.tolist()
        for seq_no in tqdm(seq_list):
            s = env.reset(seq_no=seq_no)
            ep_reward = 0
            ep_ave_max_q = 0
            ep_len = 0

            for ep_len in range(force_stop):
                a = model.get_action_dist(s)
                ob_dict = env._observe(ob_as_dict=True)
                a /= np.sum(a)
                #print(a)
                action = np.random.choice(aspace, p=a)
                chosen_actions[action] += 1
                s2, r, done, info = env.step(action)
                ep_reward += r
                replay_buffer.add(s, a, r, done, s2)
                if replay_buffer.size() >= batch_size:
                    minibatch = replay_buffer.sample_batch(batch_size)
                    pred = model.train(minibatch)
                    ep_ave_max_q += np.amax(pred)
                if done:
                    break
                s = s2

        rewards, ep_lengths, slowdowns = [], [], []
        for i in range(10):
            rew = 0
            test_env.reset(seq_no=i)
            s = test_env._observe()
            for ep_len in range(force_stop):
                a = model.get_action_dist(s)
                if FILTER_OUTPUT:
                    ob_dict = test_env._observe(ob_as_dict=True)
                    impossible_actions = get_impossible_actions(ob_dict)
                    a[impossible_actions] = -np.inf
                a = np.exp(a)
                a /= np.sum(a)
                #print(a)
                action = np.random.choice(aspace, p=a)
                #action = np.argmax(a)
                s2, r, done, info = test_env.step(action)
                rew += r
                if done:
                    break
                s = s2
            ep_lengths.append(ep_len)
            rewards.append(rew)
            slowdowns.append(get_avg_slowdown(info))
        print("[iter %d] avg episode length : %0.1f avg total rewards : %0.2f avg slowdowns: %0.2f" %
            (ite, np.mean(ep_lengths), np.mean(rewards), np.mean(slowdowns)))


if __name__ == "__main__":
    __main__()
