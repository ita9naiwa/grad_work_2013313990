import time
import functools
import tensorflow as tf

from baselines import logger
from pointer_network import pointer_networks
from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util
from baselines.common.policies import build_policy


from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.a2c.runner import Runner

from tensorflow import losses

class actor(object):
    def __init__(self,
            gamma,
            state_size, input_size, embedding_size,
            hidden_size, max_job_length,
            resource_capabilites,
            nonlinear_transform=False,
            glimpse=False):

        super(actor, self).__init__()
        self.gamma = gamma
        self.max_job_length = max_job_length
        self.state_size = state_size
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.resource_capabilites = np.array(resource_capabilites, dtype=np.float32))
        self.n_resources = len(self.resource_capabilites)
        self.nonlinear_transform = nonlinear_transform
        self.glimpse = glimpse
        self.actor = pointer_networks(
            self.state_size, self.input_size, self.embedding_size, self.hidden_size,
            self.max_job_length, self.nonlinear_transform, self.glimpse)
        self.critic = pointer_networks(
            self.state_size, self.input_size, self.embedding_size, self.hidden_size,
            self.max_job_length, self.nonlinear_transform, self.glimpse, as_critic=True)

        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=0.0001)
        self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=0.0001)
        self.critic_criterion = nn.MSELoss()

    def get_input(self, observation):
        durations = {}
        job_reqs = {}
        slot = observation['job_slot']
        max_usage = self.resource_capabilites
        current_usage = observation['machine']
        m = current_usage / max_usage

        idx = 0
        job_id_to_index = {}
        job_index_to_id = {}

        for id, (l, r) in enumerate(zip(slot['lengths'], slot['resource_vectors'])):
            if l is None:
                continue

            p = np.zeros((self.max_job_length, self.n_resources), dtype=np.float32)
            p[:l] = (r / max_usage)
            self.input_matrix[0, idx, :-1] = p.ravel()
            self.input_matrix[0, idx, -1] = 0
            durations[idx] = l
            job_reqs[idx] = torch.FloatTensor(r / max_len)
            job_id_to_index[id] = idx
            job_index_to_id[idx] = id
            idx += 1

        p = np.zeros_like(self.input_matrix[0, idx, :])
        p[-1] = 1
        self.input_matrix[0, idx, :] = p
        return (m, self.input_matrix[:, :(1 + idx), :],
                durations, job_reqs, (job_id_to_index, job_index_to_id))

    def get_action(self, ob, argmax=False):
        m, input_matrix, durations, job_reqs, (id_to_index, index_to_id) = self.get_input(observation)

        state = torch.as_tensor(m, dtype=torch.float32)
        input = torch.as_tensor(input_matrix, dtype=torch.float32)

        item_indices, _ = self.actor.get_action_(state, input, durations, job_reqs, id_to_index, index_to_id, argmax)
        return [index_to_id[idx] for idx in item_indices],  item_indices

    def get_p_a_s(self, observation, action):
        m, input_matrix, duration, job_reqs, (id_to_index, index_to_id) = self.get_input(observation)
        state = torch.as_tensor(m, dtype=torch.float32)
        input = torch.as_tensor(input_matrix, dtype=torch.float32)
        enc, (h, c) = self.encode(state, input)
        item_indices = [id_to_index[id] for id in action]
        p_a_s, entropy = self.critic.get_log_p_a_s(
            state, enc, duration, job_reqs, h, c, item_indices)

        return p_a_s, entropy


    def get_s(self, observation):
        m, input_matrix, duration, job_reqs, (id_to_index, index_to_id) = self.get_input(observation)
        state = torch.as_tensor(m, dtype=torch.float32)
        input = torch.as_tensor(input_matrix, dtype=torch.float32)
        return self.critic.get_s(state, input)

    def get_s_values(self, observations):
        return [self.get_s(ob) for ob in observations]

    def get_p_a_ss(self, observations, actions):
        p_a_ss, entropies = [], []
        for ob, action for zip(observations, actions):
            p_a_s, ent = self.get_p_a_s(ob, action)
            p_a_ss.append(p_a_s)
            entropies.append(ent)
        return p_a_ss, entropies

    def train(self, trajectory):
        observations, rewards, actions = trajectory
        list_V = self.get_s_values(observations)
        list_P_a_s, list_entropy = get_p_a_ss(self, observations, actions)
        actor_loss = 0
        critic_loss = 0
        for V,r, P, E in reversed(zip(list_V, rewards, list_P_a_s, list_entropy)):
            # modify this??...
            R = gamma * R + r
            actor_loss += ((R - V) * p_a_s + E)
            # todo:  add optimizer;
            pass














class Model(object):

    def __init__(self, policy, env, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

        sess = tf_util.get_session()
        nenvs = env.num_envs
        nbatch = nenvs*nsteps


        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            # step_model is used for sampling
            step_model = policy(nenvs, 1, sess)

            # train_model is used to train our network
            train_model = policy(nbatch, nsteps, sess)

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = train_model.pd.neglogp(A)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        # Update parameters using loss
        # 1. Get the model parameters
        params = find_trainable_variables("a2c_model")

        # 2. Calculate the gradients
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Make op for one policy and value update step of A2C
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)

        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy


        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)


def learn()