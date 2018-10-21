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

pa = Parameters()
env = Env(pa, end='all_done')
sess = tf.Session()
ob = env.reset()
ob = flatten(ob)
state_dim = len(ob)
action_dim = pa.num_res + 1
discount_factor = 1.00
batch_size = 64
num_episodes = 1000
pa.episode_max_length = episode_max_length = 200
render = True
buffer_size = 100000
actor_lr = 0.0001
critic_lr = 0.001
tau = 0.001
seed = 1234


def __main__():
    model = ddpg.DDPG(sess, action_dim, len(ob), actor_lr, critic_lr, tau=tau)
    sess.run(tf.initializers.global_variables())
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
    replay_buffer = ReplayBuffer(buffer_size)
    reward = Reward(0.1, discount_factor)
    slowdowns = []
    for current_ep in range(num_episodes):
        s = env.reset()
        s = flatten(s)
        ep_reward = 0
        episode_buffer = np.empty((0, 5), float)
        for ep_len in range(episode_max_length):
            a = model.get_action_dist(s)

            if(current_ep <= 15):
                a += actor_noise()

            action = np.argmax(a)
            s2, r, done, info = env.step(action)
            s2 = flatten(s)
            ep_reward += r
            episode_buffer = np.append(episode_buffer, [[s, a, r, done, s2]], axis=0)

            if replay_buffer.size() >= batch_size:
                minibatch = replay_buffer.sample_batch(batch_size)
                model.train(minibatch)

            if done:
                break
            s = s2

        episode_buffer = reward.discount(episode_buffer)
        for step in episode_buffer:
            replay_buffer.add(step[0], step[1], step[2], step[3], step[4])

        slowdown = get_avg_slowdown(info)
        slowdowns.append(slowdown)
        print("[episode %d] episode length : %d, slowdown : %0.2f, rew_sum : %0.2f" % (current_ep, ep_len, slowdown, ep_reward))

if __name__ == "__main__":
    __main__()
