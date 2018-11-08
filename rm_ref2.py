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
episode_max_length = 100
num_slots = 5
max_ep_size = 400
env = mao.ClusteringEnv(p_job_arrival=0.6, observation_mode='image',
        episode_size=episode_max_length, force_stop=max_ep_size, num_slots=num_slots,
        n_resource_slot_capacities=(10, 10))
sess = tf.Session()
ob = env.reset()
state_dim = np.prod(ob.shape)
action_dim = env.action_space.n
discount = 0.9
batch_size = 8
num_episodes = 10000
render = True
buffer_size = 2000
lr = 0.001
seed = 1234
batch_size = 16
model = rl.model(sess, state_dim, action_dim, lr, network_widths=[200, 30])

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

    sess.run(tf.initializers.global_variables())
    replay_buffer = ReplayBuffer(buffer_size)
    moving_mean = np.zeros(shape=(100, max_ep_size))
    sd_prev = np.ones(50)
    for current_ep in range(num_episodes):
        seq_no = np.random.randint(0, 50)
        s = env.reset(seq_no=seq_no)
        s = flatten(s)
        ep_reward = 0
        ep_ave_max_q = 0
        ep_len = 0
        episode_buffer = np.empty((0, 5), float)
        reward = Reward(0.1, discount)

        #for ep_len in tqdm(range(max_ep_size)):
        for ep_len in range(max_ep_size):
            a = model.get_action_dist(s)
            ob_dict = env._observe(ob_as_dict=True)
            impossible_actions = get_impossible_actions(ob_dict)
            a[impossible_actions] = -np.inf
            a = np.exp(a)
            a /= np.sum(a)
            #print(a)
            action = np.random.choice(aspace, p=a)
            chosen_actions[action] += 1
            s2, r, done, info = env.step(action)
            s2 = flatten(s2)

            ep_reward += r
            episode_buffer = np.append(episode_buffer, [[s, action, r, done, s2]], axis=0)
            #replay_buffer.add(s, action, (r - moving_mean[seq_no][ep_len]), done, s2)
            if moving_mean[seq_no][ep_len] == 0:
                moving_mean[seq_no][ep_len] = r
            else:
                moving_mean[seq_no][ep_len] = moving_mean[seq_no][ep_len] * 0.9 + r * 0.1
            #print(done)
            if done:
                #episode_buffer = reward.discount(episode_buffer)
                #for step in episode_buffer:
                #    replay_buffer.add(step[0], step[1], step[2], step[3], step[4])
                break
            s = s2

        for i, step in enumerate(episode_buffer):
            if moving_mean[seq_no][i] == 0:
                moving_mean[seq_no][i] = step[2]

            replay_buffer.add(step[0], step[1], step[2] - moving_mean[seq_no][i], step[3], step[4])
            moving_mean[seq_no][i] = step[2] * 0.1 + 0.9* moving_mean[seq_no][i]
        minibatch = replay_buffer.sample_batch(batch_size)
        pred = model.train(minibatch[0], minibatch[1], minibatch[2])
        replay_buffer.clear()


        slowdown = get_avg_slowdown(info)
        print(chosen_actions)
        print("[episode %d] average episode length : %d" % (current_ep, ep_len), "episode reward : %f, Slowdown: %0.2f" % (ep_reward, slowdown), "rew_prev : %0.2f , rew_now : %0.2f, dec : %0.2f" % (sd_prev[seq_no], ep_reward, ep_reward / sd_prev[seq_no]))
        sd_prev[seq_no] = ep_reward

if __name__ == "__main__":
    __main__()
