import tensorflow as tf
import gym
import numpy as np
import src.envs.Mao as mao
from time import sleep
import src.models.ddpg as ddpg
from src.models.replay_buffer import ReplayBuffer
from src.noise import OrnsteinUhlenbeckActionNoise
from src.summary import summary
env = mao.ClusteringEnv()
ob = env.reset()
sess = tf.Session()
sess = tf.Session()
state_dim = np.prod(ob.shape)
action_dim = env.action_space.n
discount = 0.99
batch_size = 64
num_episodes = 1000
max_episode_length = 200
render = True
buffer_size = 100000
actor_lr = 0.0001
critic_lr = 0.001
tau = 0.001
seed = 1234


model = ddpg.DDPG(sess, action_dim, state_dim, actor_lr, critic_lr, tau=tau)
sess.run(tf.initializers.global_variables())
def flatten(m, a=state_dim):
    return np.reshape(m, newshape=(state_dim,))

for i_episode in range(num_episodes):
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
    replay_buffer = ReplayBuffer(buffer_size)
    s = np.reshape(env.reset(), newshape=(state_dim,))
    ep_reward = 0.0
    ep_ave_max_q = 0.0
    #print("episode %d" % i_episode)
    for curr_ep_len in range(max_episode_length):
        #env.render()
        a = model.get_action_dist(s)
        if(i_episode <= 30):
            a += actor_noise()
        action = np.argmax(a)
        #print(a)
        s2, reward, done, info = env.step(action)
        ep_reward += reward
        s2 = flatten(s2)
        replay_buffer.add(s, a, reward, done, s2)
        if replay_buffer.size() >= batch_size:
            minibatch = replay_buffer.sample_batch(batch_size)
            pred = model.train(minibatch)
            #print(pred)
            ep_ave_max_q += np.amax(pred)
        if done:
            break
        #print("chosen action :", action)
        #print("reward :", reward)
        """
        if True:
            print("action", action)
            print("observed", s)
            print("reward", reward)
            print("done?", done)
        """
        s = s2
    print("[episode %d] avg episode reward : %0.2f, avg slowdown %0.2f" %
            (i_episode,  ep_reward / float(curr_ep_len), env._get_avg_slowdown()))





#        if done:
#            print("Episode finished after {} timesteps".format(t+1))
#            break
