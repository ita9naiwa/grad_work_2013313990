import test_base
import tensorflow as tf
from models.RL_model import actor_network
from models.baselines import Fifo_model
from models.baselines import random_action_model



if __name__ == "__main__":
    sess = tf.Session()
    my_model = actor_network(sess,
        action_dim=3,
        input_width=2,
        input_height=2,
        learning_rate=0.01,
        tau=0.01)

    actions = [1,2]
    states = [[1,2,3,4],[5,6,7,8]]
    values = [3.0, 2.1]


    ret = my_model.train(states, actions, values)
    print(ret)