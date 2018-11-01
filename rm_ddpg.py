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
episode_max_length = 50
num_slots = 10
max_ep_size = 100
env = mao.ClusteringEnv(p_job_arrival=0.7, observation_mode='image',
        episode_size=50, force_stop=100, num_slots=num_slots,
        n_resource_slot_capacities=(20, 20))
sess = tf.Session()
ob = env.reset()
state_dim = np.prod(ob.shape)
action_dim = env.action_space.n
discount = 0.9
batch_size = 16
num_episodes = 1000
render = True
buffer_size = 5000
actor_lr = 1e-5
critic_lr = 1e-5
tau = 0.001
seed = 1234
model = ddpg.DDPG(sess, action_dim, state_dim, actor_lr, critic_lr, tau=tau, use_softmax=True)

def flatten(m, a=state_dim):
    return np.reshape(m, newshape=(state_dim,))

aspace = np.arange(action_dim)
def __main__():

    sess.run(tf.initializers.global_variables())
    replay_buffer = ReplayBuffer(buffer_size)
    for current_ep in range(num_episodes):
        s = env.reset(reset_scenario=False)
        s = flatten(s)
        ep_reward = 0
        ep_ave_max_q = 0
        ep_len = 0

        for ep_len in tqdm(range(max_ep_size)):
            a = model.get_action_dist(s)
            action = np.random.choice(aspace, p=a)
            s2, r, done, info = env.step(action)
            s2 = flatten(s2)
            ep_reward += r
            #episode_buffer = np.append(episode_buffer, [[s, a, r, done, s2]], axis=0)
            replay_buffer.add(s, a, r, done, s2)
            if replay_buffer.size() >= batch_size:
                minibatch = replay_buffer.sample_batch(batch_size)
                pred = model.train(minibatch)
                ep_ave_max_q += np.amax(pred)
            if done:
                #episode_buffer = reward.discount(episode_buffer)
                #for step in episode_buffer:
                    #replay_buffer.add(step[0], step[1], step[2], step[3], step[4])
                break
            s = s2

        slowdown = get_avg_slowdown(info)
        print("[episode %d] average episode length : %d" % (current_ep, ep_len), "episode reward : %f, Slowdown: %0.2f" % (ep_reward, slowdown))

if __name__ == "__main__":
    __main__()
