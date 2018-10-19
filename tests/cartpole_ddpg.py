import test_base

import src.models.ddpg as ddpg
from src.models.replay_buffer import ReplayBuffer
from src.noise import OrnsteinUhlenbeckActionNoise
from src.summary import summary
import numpy as np
import tensorflow as tf
import gym
import scipy.signal

env = gym.make('CartPole-v1')
sess = tf.Session()
ob = env.reset()
state_dim = 4
action_dim = env.action_space.n
discount = 0.99
batch_size = 100
num_episodes = 1000
episode_max_length=500
render = True
buffer_size = 100000
actor_lr = 0.0001
critic_lr = 0.001
tau = 0.001
seed = 1234

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
    model = ddpg.DDPG(sess, action_dim, len(ob), actor_lr, critic_lr, tau=tau)
    sess.run(tf.initializers.global_variables())

    summary_writer = summary(sess, "./ddpg_log", episode_len=int, qmax=float)

    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
    replay_buffer = ReplayBuffer(buffer_size)
    episode_buffer = np.empty((0, 5), float)
    reward = Reward(0.1, discount)
    for current_ep in range(num_episodes):
        s = env.reset()
        ep_reward = 0
        ep_ave_max_q = 0
        ep_len = 0
        episode_buffer = np.empty((0,5), float)
        for ep_len in range(episode_max_length):

            a = model.get_action_dist(s)
            if(current_ep <= 200):
                a += actor_noise()
            else:
                env.render()
            action = np.argmax(a)
            s2, r, done, info = env.step(action)
            ep_reward += r
            episode_buffer = np.append(episode_buffer, [[s, a, r, done, s2]], axis=0)

            if replay_buffer.size() >= batch_size:
                minibatch = replay_buffer.sample_batch(batch_size)
                pred = model.train(minibatch)
                ep_ave_max_q += np.amax(pred)

            if done:
                episode_buffer = reward.discount(episode_buffer)
                for step in episode_buffer:
                    replay_buffer.add(step[0], step[1], step[2], step[3], step[4])
                break
            s = s2
        summary_writer.write_log(episode_len=ep_len, qmax=ep_ave_max_q / float(ep_len))
        print("[episode %d] average episode length : %d" % (current_ep, ep_len), "episode reward : %d, Qmax : %0.2f" % (ep_reward, float(ep_ave_max_q / float(ep_len))))

if __name__ == "__main__":
    __main__()
