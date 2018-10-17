import tensorflow as tf
import gym
import src.models.ddpg as ddpg
env = gym.make('CartPole-v0')

ob = env.reset()
print(env.action_space)
model = ddpg.DDPG(tf.Session(), 2, 4, 0.01, 0.1)


while True:
    print(ob)
    #env.render()
    action = env.action_space.sample()
    ob, rew, done, info = env.step(action)
    if done:
        break
