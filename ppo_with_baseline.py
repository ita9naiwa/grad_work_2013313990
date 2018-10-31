import pickle
from tqdm import tqdm
import tensorflow as tf
import gym
import numpy as np
from time import sleep
import src.models.REINFORCE as reinforce
import src.models.REINFORCE_PPO as reinforce
from src.models.buffer import ReplayBuffer
from src.summary import summary
from src.utils import *
from src.deeprm.parameters import Parameters, Dist
from src.deeprm.environment import Env
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

pa = Parameters()
pa.num_ex = 50
env = Env(pa)
ob = env.reset()
ob = flatten(ob)
del env
sess = tf.Session()
state_dim = len(ob)
action_dim = pa.num_nw + 1
discount_factor = 1.00
num_episodes = 1000

lr = 0.0003
update_step = 50
print(ob.shape)

chosen_action_dist = [0 for _ in range(pa.res_slot + 1)]

def calc_entropy(p):
    p = p + 0.0001
    ret = -np.sum(np.log(p) * p)
    if np.isinf(ret):
        return 0
    else:
        return ret

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

nw_len_seqs, nw_size_seqs = generate_sequence_work(pa, seed=42)

envs = []
for ex in range(pa.num_ex):
    env = Env(pa, nw_len_seqs=nw_len_seqs, nw_size_seqs=nw_size_seqs, end='all_done')
    env.seq_no = ex
    envs.append(env)
with open('test_env.pickle', 'rb') as f:
    te_env = pickle.load(f)

def flatten(m, a=state_dim):
    return np.reshape(m, newshape=(state_dim,))

model = reinforce.model(sess, state_dim, action_dim, lr,
                network_widths=[30],
                update_step=update_step)
sess.run(tf.initializers.global_variables())
action_space = np.arange(action_dim)

def get_ith_handle(model, idx):
    env = envs[idx]
    job_buffer = []
    rewards = []
    slowdowns = []
    ep_lens = []


    for ep in range(pa.num_seq_per_batch):
        job, ep_reward, slowdown, ep_len = get_traj(env, model)
        job = (job[0], job[1], job[2], job[3], None)
        job_buffer.append(job)
        rewards.append(ep_reward)
        slowdowns.append(slowdown)
        ep_lens.append(ep_len)

    n_samples = len(job_buffer)
    max_job_length = np.max([len(x[0]) for x in job_buffer])
    baseline = np.zeros(shape=(n_samples, max_job_length), dtype=np.float32)
    ep_lengths = []

    for i in range(n_samples):
        current_length = len(job_buffer[i][3])
        ep_lengths.append(current_length)
        baseline[i, :current_length] = job_buffer[i][3]

    baseline = np.mean(baseline, axis=0)
    ret = []
    for ep in range(pa.num_seq_per_batch):
        reward_list = job_buffer[ep][3]
        adv = []
        for t in range(len(reward_list)):
            adv.append(reward_list[t] - baseline[t])
        ret.append((job_buffer[ep][0], job_buffer[ep][1], job_buffer[ep][2], job_buffer[ep][3], adv))

    return ret, baseline, rewards, slowdowns, ep_lens

def get_traj(env, model):
    ep_reward = 0.0
    s = env.reset()
    s = flatten(s)
    list_s = []
    list_a = []
    list_r = []
    list_y = []
    for ep_len in range(pa.episode_max_length):

        a = model.get_action_dist(s)
        csprob_n = np.cumsum(a)
        action = (csprob_n > np.random.rand()).argmax()
        chosen_action_dist[action] = chosen_action_dist[action] + 1
        s2, r, done, info = env.step(action)
        s2 = flatten(s2)
        list_s.append(s)
        list_a.append(action)
        list_r.append(r)
        ep_reward += r
        if done:
            break
        s = s2

    slowdown = get_avg_slowdown(info)
    list_y = discount(list_r, discount_factor)
    return (list_s, list_a, list_r, list_y), ep_reward, slowdown, ep_len


for i_episode in range(num_episodes):
    s = np.reshape(env.reset(), newshape=(state_dim,))
    ep_reward = 0.0
    ep_ave_max_q = 0.0
    #print("episode %d" % i_episode)
    job_buffers = []
    baselines = []
    rewards = []
    futures = []
    slowdowns = []
    ep_lens = []
    with ThreadPoolExecutor(max_workers=12) as exec:
        for ex in range(pa.num_ex):
            futures.append(exec.submit(get_ith_handle, model, ex))

    concurrent.futures.wait(futures)
    advs = []
    for i in range(pa.num_ex):
        job_buffer, baseline, rew, sd, eplen = futures[i].result()
        job_buffers.append(job_buffer)
        baselines.append(baseline)
        rewards.append(rew)
        slowdowns.append(sd)
        ep_lens.append(eplen)

    # compute baseline
    ss = []
    aa = []
    vv = []
    for t in range(pa.episode_max_length):
        for ep in range(pa.num_seq_per_batch):
            for ex in range(pa.num_ex):
                if len(job_buffers[ex][ep][0]) <= t:
                    continue
                s = job_buffers[ex][ep][0][t]
                a = job_buffers[ex][ep][1][t]
                adv = job_buffers[ex][ep][4][t]

            ss.append(s)
            aa.append(a)
            vv.append(adv)
    model.train(np.array(ss), np.array(aa), np.array(vv))

    print(
        "[episode %d] avg episode length %0.2f avg slowdown %0.2f, avg reward %0.2f" %
        (i_episode, np.mean(ep_lens), np.mean(slowdowns), np.mean(rewards)))
    if(i_episode+1) % 10 == 0:
        entropies = []
        slowdowns = []
        for ex in range(pa.num_ex):
            s = te_env.reset()
            s = flatten(s)
            te_env.seq_no = ex
            for ep_len in range(pa.episode_max_length):
                a = model.get_action_dist(s)
                entropies.append(calc_entropy(a))
                action = np.random.choice(action_space, p=a)
                s2, r, done, info = te_env.step(action)
                s2 = flatten(s2)
                if done:
                    break
                s = s2
            slowdown = get_avg_slowdown(info)
            slowdowns.append(slowdown)
        print("[test res at %d ]\tAvg slowdown of test dataset: %0.2f, Avg entropy %0.2f" % (i_episode, np.mean(slowdowns), np.mean(entropies)))
