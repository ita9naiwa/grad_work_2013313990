from tqdm import tqdm
import src.models.REINFORCE as rl
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
episode_max_length = 50
max_ep_size = 100
num_slots = 5

env = mao.ClusteringEnv(p_job_arrival=0.7, observation_mode='image',
        episode_size=50, force_stop=100, num_slots=num_slots,
        n_resource_slot_capacities=(20, 20))

sess = tf.Session()
ob = env.reset()
state_dim = np.prod(ob.shape)
action_dim = env.action_space.n
discount = 0.8
num_episodes = 1000
render = True
buffer_size = 5000
lr = 0.0001
seed = 1234
num_job_sets = 100
batch_size = 50
model = rl.model(sess, state_dim, action_dim, lr, network_widths=[50, 50, 40])

def flatten(m, a=state_dim):
    return np.reshape(m, newshape=(state_dim,))

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
            ep_batch[i, 2] = discounted[i]

        return ep_batch
from time import sleep

aspace = np.arange(action_dim)
def __main__():
    sess.run(tf.initializers.global_variables())
    sigma = np.diag(0.3 * np.ones(action_dim, dtype=np.float32))

    for current_job_cnt in range(num_job_sets):
        job_buffer = []
        rewards = []
        slowdowns = []
        for current_ep in range(batch_size):
            ep_reward = 0.0
            ep_ave_max_q = 0.0
            s = env.reset(reset_scenario=False)
            s = flatten(s)
            list_s = []
            list_a = []
            list_r = []
            list_y = []
            for ep_len in range(max_ep_size):
                a = model.get_action_dist(s)
                action = np.random.choice(aspace, p=a)
                s2, r, done, info = env.step(action)
                s2 = flatten(s2)
                list_s.append(s)
                list_a.append(action)
                list_r.append(r)
                ep_reward += r
                if done or ep_len == episode_max_length-1:
                    disc_vec = np.array(list_r, dtype=np.float32)
                    for i in range(1, len(list_y)):
                        disc_vec[i:] *= discount
                    for i in range(len(list_r)):
                        y_i = np.sum(disc_vec[i:]) / (discount ** i)
                        list_y.append(y_i)
                    job_buffer.append((list_s, list_a, list_r, list_y))
                    rewards.append(ep_reward)
                    slowdowns.append(get_avg_slowdown(info))
                    break
                s = s2

        # compute baseline
        n_samples = len(job_buffer)
        max_job_length = np.max([len(x[0]) for x in job_buffer])
        baseline = np.zeros(shape=(n_samples, max_job_length), dtype=np.float32)
        ep_lengths = []
        for i in range(n_samples):
            current_length = len(job_buffer[i][3])
            ep_lengths.append(current_length)
            baseline[i, :current_length] = job_buffer[i][3]
        baseline = np.mean(baseline, axis=0)
        ss = []
        aa = []
        vv = []

        for t in range(max_job_length):
            for i in range(n_samples):
                if ep_lengths[i] <= t:
                    continue
                s = job_buffer[i][0][t]
                a = job_buffer[i][1][t]
                y = job_buffer[i][3][t]
                b = baseline[t]
                var = y - b
                ss.append(s)
                aa.append(a)
                vv.append(var)

        loss = model.train(np.array(ss), np.array(aa), np.array(vv))
        print(loss)
        env.reset(reset_scenario=True)
        print("[jobset %d] avg episode length : %0.2f" % (current_job_cnt, np.mean(ep_lengths)), "avg episode reward : %0.2f" % np.mean(rewards), "slowdown %0.2f" % np.mean(slowdowns))

if __name__ == "__main__":
    __main__()
