import pickle

import src.models.ddpg as ddpg
from src.models.buffer import ReplayBuffer
from src.noise import OrnsteinUhlenbeckActionNoise
from src.summary import summary
import numpy as np
import tensorflow as tf
import gym
import scipy.signal
from src.models.buffer import ReplayBuffer
from src.summary import summary
from src.utils import *
from src.deeprm.parameters import Parameters
from src.deeprm.environment import Env

def generate_sequence_work(pa, seed=42):
    np.random.seed(seed)
    simu_len = pa.simu_len * pa.num_ex
    nw_dist = pa.dist.bi_model_dist
    nw_len_seq = np.zeros(simu_len, dtype=int)
    nw_size_seq = np.zeros((simu_len, pa.num_res), dtype=int)
    for i in range(simu_len):
        if np.random.rand() < pa.new_job_rate:  # a new job comes
            nw_len_seq[i], nw_size_seq[i, :] = nw_dist()
    nw_len_seq = np.reshape(nw_len_seq, [pa.num_ex, pa.simu_len])
    nw_size_seq = np.reshape(nw_size_seq, [pa.num_ex, pa.simu_len, pa.num_res])
    return nw_len_seq, nw_size_seq

pa = Parameters()
pa.compute_dependent_parameters()
nw_len_seqs, nw_size_seqs = generate_sequence_work(pa, seed=35)
# rew type list:
# delay, mine,
env = Env(pa, nw_len_seqs=nw_len_seqs, nw_size_seqs=nw_size_seqs, end='all_done', reward_type='mine')
with open('test_env.pickle', 'rb') as f:
    te_env = pickle.load(f)

sess = tf.Session()
ob = env.reset()
ob = flatten(ob)
state_dim = len(ob)
action_dim = pa.num_nw + 1
print("action_dim", action_dim)
discount_factor = 0.99
batch_size = 1024
num_episodes = 2001
render = True
buffer_size = 100000
actor_lr =  0.0001
critic_lr = 0.001
tau = 0.001
seed = 1234
small_constant = 0.1
def calc_entropy(p):
    p = p + 0.0001
    ret = -np.sum(np.log(p) * p)
    if np.isinf(ret):
        return 0
    else:
        return ret

target_step = 30

def __main__():
    model = ddpg.DDPG(sess, action_dim, len(ob), discount_factor, actor_lr, critic_lr, tau=tau, use_softmax=True)
    sess.run(tf.initializers.global_variables())
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
    replay_buffer = ReplayBuffer(buffer_size)
    #sup_buffer = ReplayBuffer(buffer_size)
    reward = Reward(small_constant, discount_factor)
    slowdowns = []
    action_space = np.arange(action_dim)
    for current_ep in range(num_episodes):
        s = env.reset()

        env.seq_no = (env.seq_no + 1) % env.pa.num_ex
        s = flatten(s)
        ep_reward = 0
        #episode_buffer = np.empty((0, 5), float)
        ents = []

        chosen_action_dist = np.zeros(shape=(pa.num_nw+1, ))
        for ep_len in range(pa.episode_max_length):

            a = model.get_action_dist(s)
            #print(a)
            ents.append(calc_entropy(a))
            chosen_action_dist += a
            action = np.random.choice(action_space, p=a)
            s2, r, done, info = env.step(action)
            chosen_action_dist[action] = chosen_action_dist[action] + 1
            s2 = flatten(s2)
            ep_reward += r
            #episode_buffer = np.append(episode_buffer, [[s, a, r, done, s2]], axis=0)
            replay_buffer.add(s, a, small_constant * r, done, s2)

            #if replay_buffer.count >= batch_size:
            if current_ep >= env.pa.num_ex:
                minibatch = replay_buffer.sample_batch(batch_size)
                model.train(minibatch)

            if done:
                #episode_buffer = reward.discount(episode_buffer)
                #for step in episode_buffer:
                #    replay_buffer.add(step[0], step[1], step[2], step[3], step[4])
                break
            s = s2

        #if(current_ep + 1) % 20 == 0:
        #    replay_buffer.clear()

        #slowdown = get_avg_slowdown(info)
        #slowdowns.append(slowdown)
        print(chosen_action_dist)
        print("[episode %d], mean entropcy: %0.2f" % (current_ep, np.mean(ents)))
        #print("[episode %d] episode length : %d, slowdown : %0.2f, rew_sum : %0.2f, mean entropy : %0.2f" % (current_ep, ep_len, slowdown, ep_reward, np.mean(ents)))


        entropies = []
        if((1+current_ep) % 100 == 0):
            slowdowns = []
            for ex in range(pa.num_ex):
                s = te_env.reset()
                s = flatten(s)
                te_env.seq_no = ex
                for ep_len in range(pa.episode_max_length):
                    a = model.get_action_dist(s)
                    entropies.append(calc_entropy(a))
                    action = np.random.choice(action_space, p=a)
                    #action = np.argmax(a)
                    s2, r, done, info = te_env.step(action)
                    s2 = flatten(s2)
                    if done:
                        break
                    s = s2
                slowdown = get_avg_slowdown(info)
                slowdowns.append(slowdown)

            print("[test res at %d ]\tAvg slowdown of test dataset: %0.2f, Avg entropy %0.2f" % (current_ep, np.mean(slowdowns), np.mean(entropies)))

if __name__ == "__main__":
    __main__()
