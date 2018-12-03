import numpy as np
from .pointer_network import pointer_networks
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
def discount_reward(r, gamma, final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class model(object):
    def __init__(self,
            gamma,
            state_size, input_size, embedding_size,
            hidden_size, max_job_length,
            resource_capabilites,
            nonlinear_transform=False,
            glimpse=False):

        super(model, self).__init__()
        self.gamma = gamma
        self.max_job_length = max_job_length
        self.state_size = state_size
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.resource_capabilites = np.array(resource_capabilites, dtype=np.float32)
        self.n_resources = len(self.resource_capabilites)
        self.nonlinear_transform = nonlinear_transform
        self.glimpse = glimpse
        self.actor = pointer_networks(
            self.state_size, self.input_size, self.embedding_size, self.hidden_size,
            self.max_job_length, self.nonlinear_transform, self.glimpse)
        self.critic = pointer_networks(
            self.state_size, self.input_size, self.embedding_size, self.hidden_size,
            self.max_job_length, self.nonlinear_transform, self.glimpse, as_critic=True)
        self.input_matrix = np.empty(shape=(1, 128, self.input_size), dtype=float)

        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=1e-3)
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
            job_reqs[idx] = torch.FloatTensor(r / self.resource_capabilites)
            job_id_to_index[id] = idx
            job_index_to_id[idx] = id
            idx += 1

        p = np.zeros_like(self.input_matrix[0, idx, :])
        p[-1] = 1
        self.input_matrix[0, idx, :] = p
        return (m, self.input_matrix[:, :(1 + idx), :],
                durations, job_reqs, (job_id_to_index, job_index_to_id))

    def get_action(self, ob, argmax=False):
        m, input_matrix, durations, job_reqs, (id_to_index, index_to_id) = self.get_input(ob)

        state = torch.as_tensor(m, dtype=torch.float32)
        input = torch.as_tensor(input_matrix, dtype=torch.float32)

        item_indices, _ = self.actor.get_action_(state, input, durations, job_reqs, id_to_index, index_to_id, argmax)
        return [index_to_id[idx] for idx in item_indices],  item_indices

    def get_logpas(self, observation, action):
        m, input_matrix, duration, job_reqs, (id_to_index, index_to_id) = self.get_input(observation)
        state = torch.as_tensor(m, dtype=torch.float32)
        input = torch.as_tensor(input_matrix, dtype=torch.float32)
        enc, (h, c) = self.critic.encode(state, input)
        item_indices = [id_to_index[id] for id in action]
        p_a_s, entropy = self.critic.get_log_p_a_s(
            state, enc, duration, job_reqs, h, c, item_indices)

        return p_a_s, entropy


    def get_s(self, observation):
        m, input_matrix, durations, job_reqs, (id_to_index, index_to_id) = self.get_input(observation)
        state = torch.as_tensor(m, dtype=torch.float32)
        input = torch.as_tensor(input_matrix, dtype=torch.float32)
        return self.critic.get_s(state, input, durations)

    def get_s_values(self, observations):
        return torch.cat([self.get_s(ob) for ob in observations], 0).squeeze(1)

    def get_logpass(self, observations, actions):
        p_a_ss, entropies = [], []
        #print("observations", observations)
        for ob, action in zip(observations, actions):
            p_a_s, ent = self.get_logpas(ob, action)
            p_a_ss.append(p_a_s)
            entropies.append(ent)
        p_a_ss = torch.stack(p_a_ss, 0)
        ent = torch.stack(entropies, 0)
        return p_a_ss, ent

    def train(self, obs, rews, acts):
        observations, rewards, actions = obs, rews, acts
        pred_V = self.get_s_values(observations)
        # TODO: 마지막 값이 0이 아니게 되어도 될 수 있게 'ㅅ'...
        # 모든 스텝이 끝나서가 아니라, multi-n-step으로 업데이트하게 고쳐야 함.

        Returns = Variable(torch.Tensor(discount_reward(rewards, 0.99, 0)))
        logPas, Entropy = self.get_logpass(observations, actions)

        critic_loss = torch.mean((pred_V - Returns)**2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        adv = Returns - pred_V.detach()
        actor_loss = torch.mean(-(adv * logPas) + Entropy)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.detach().numpy(), critic_loss.detach().numpy()
