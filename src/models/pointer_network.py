import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle

def get_entorpy(v):
    v += 0.0001
    return -np.sum(np.log(v) * v)


class Attention(nn.Module):
    def __init__(self, hidden_size, use_tanh=False, use_cuda=False):
        super(Attention, self).__init__()
        self.W_enc = nn.Parameter(Variable(torch.FloatTensor(hidden_size, hidden_size)))
        self.W_ref = nn.Parameter(Variable(torch.FloatTensor(hidden_size, hidden_size)))
        self.V = nn.Parameter(Variable(torch.FloatTensor(hidden_size)))

        self.V.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))
        self.W_enc.data.uniform_(-(1. / math.sqrt(hidden_size)) , 1. / math.sqrt(hidden_size))
        self.W_ref.data.uniform_(-(1. / math.sqrt(hidden_size)) , 1. / math.sqrt(hidden_size))


    def forward(self, enc, ref):
        seq_len = enc.size(1)
        """
        Args:
            enc: [batch_size x seq_len x hidden_size]
            ref: [batch_size x hidden_size]
        Output:
        """
        Wenc = torch.einsum("ak,bjk->bja", [self.W_enc, enc])
        Wref = torch.einsum("ak,bk->ba", [self.W_ref, ref]).unsqueeze(1).repeat(1, seq_len, 1)
        # [batch_size x seq_len x hidden_size] reference vector multiplied by w_enc
        W = torch.einsum("k,ijk->ij", [self.V, torch.tanh(Wenc + Wref)])
        return W


class input_transform(nn.Module):
    def __init__(self, input_size, embedding_size=32, nonlinear=False):
        super(input_transform, self).__init__()
        self.enc = nn.Parameter(Variable(torch.FloatTensor(input_size, embedding_size)))
        var = math.sqrt(input_size + embedding_size)
        self.enc.data.uniform_(-(1. / var) , 1. / var)
        self.nonlinear = nonlinear
        if nonlinear is True:
            self.enc2 = nn.Parameter(Variable(torch.FloatTensor(embedding_size, embedding_size)))
            self.bias1 = nn.Parameter(Variable(torch.FloatTensor(embedding_size)))
            self.bias2 = nn.Parameter(Variable(torch.FloatTensor(embedding_size)))
            self.enc2.data.uniform_(-(1. /var), 1. / var)
            self.bias1.data.uniform_(-0.001, 0.001)
            self.bias2.data.uniform_(-0.001,0.001)

    def forward(self, x):
        """
        Args:
            x: [batch_size x seq_len x input_size]
        """
        embedded = torch.einsum("ji,bsj->bsi", [self.enc, x])
        if self.nonlinear is True:
            embedded = torch.einsum("bsi,i->bsi", [embedded, self.bias1])
            embedded = torch.einsum("ji,bsj->bsi", [self.enc2, torch.tanh(embedded)])
            embedded = torch.tanh(torch.einsum("bsi,i->bsi", [embedded, self.bias2]))
        return embedded

class state_transform(nn.Module):
    def __init__(self, input_size, embedding_size=32, nonlinear=False):
        super(state_transform, self).__init__()
        self.enc = nn.Parameter(Variable(torch.FloatTensor(input_size, embedding_size)))
        var = math.sqrt(input_size + embedding_size)
        self.enc.data.uniform_(-(1. / var) , 1. / var)
        self.nonlinear = nonlinear
        if nonlinear is True:
            self.enc2 = nn.Parameter(Variable(torch.FloatTensor(embedding_size, embedding_size)))
            self.bias1 = nn.Parameter(Variable(torch.FloatTensor(embedding_size)))
            self.bias2 = nn.Parameter(Variable(torch.FloatTensor(embedding_size)))
            self.enc2.data.uniform_(-(1. / var), 1. / var)
            self.bias1.data.uniform_(-0.001, 0.001)
            self.bias2.data.uniform_(-0.001, 0.001)

    def forward(self, x):
        """
        Args:
            x: [batch_size x seq_len x input_size]
        """
        embedded = torch.einsum("ji,sj->si", [self.enc, x])
        if self.nonlinear is True:
            embedded = torch.einsum("si,i->si", [embedded, self.bias1])
            embedded = torch.einsum("ji,sj->si", [self.enc2, torch.tanh(embedded)])
            embedded = torch.tanh(torch.einsum("si,i->si", [embedded, self.bias2]))
        return embedded


class pointer_networks(nn.Module):
    def __init__(self,
            state_size,
            input_size,
            embedding_size,
            hidden_size,
            max_job_length,
            nonlinear_transform=False,
            glimpse=False,
            use_tanh=True,
            as_critic=False):
        super(pointer_networks, self).__init__()

        self.MASK_CONSTANT = 1000.0
        self.max_job_length = max_job_length
        self.state_size = state_size
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.nonlinear_transform = nonlinear_transform
        self.glimpse = glimpse
        self.use_tanh = use_tanh
        self.as_critic = as_critic


        if self.as_critic is True:
            self.nn1 = nn.Linear(self.hidden_size, self.hidden_size)
            self.nn2 = nn.Linear(self.hidden_size, 1)

        self.pointer = Attention(self.hidden_size, self.embedding_size)
        self.input_transform = input_transform(
            self.input_size, self.embedding_size,
            nonlinear=nonlinear_transform)
        self.state_transform = input_transform(
            self.state_size, self.hidden_size,
            nonlinear=self.nonlinear_transform)



        self.encoder = nn.LSTM(
            #self.input_size,
            self.embedding_size,
            hidden_size,
            num_layers=1,
            batch_first=True)  # change num layer needed?
        self.decoder = nn.LSTM(
            self.embedding_size,
            hidden_size,
            num_layers=1,
            batch_first=True)

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.decoder_start_input.data.uniform_(
            -(1. / math.sqrt(self.hidden_size)), 1. / math.sqrt(self.hidden_size))
        self.decoder_terminate_input = nn.Parameter(Variable(torch.FloatTensor(self.hidden_size)))
        self.decoder_terminate_input.data.uniform_(-(1. / math.sqrt(self.hidden_size)), 1. / math.sqrt(self.hidden_size))
        self.supervised_criterion = nn.CrossEntropyLoss(reduction='none')
        self.rl_criterion = None
        self.input_matrix = np.empty(shape=(1, 128, self.input_size), dtype=float)

    def encode(self, states, inputs):
        batch_size = inputs.size(0)

        # no embedding
        input_embedded = self.input_transform(inputs)
        #input_embedded = inputs
        out, (h, c) = self.encoder(input_embedded)
        return out, (h, c)

    def get_input(self, observation):
        slot = observation['job_slot']
        max_len = np.array([100, 100, 100], dtype=np.float32)
        m = (observation['machine'] / max_len)
        durations = dict()
        job_reqs = dict()
        #m *= 1.5
        idx = 0
        job_id_to_index = {}
        job_index_to_id = {}

        for id, (l, r) in enumerate(zip(slot['lengths'], slot['resource_vectors'])):
            if l is None:
                continue
            p = np.zeros((20, 3), dtype=np.float32)
            p[:l] = (r / max_len)
            #print(p)
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
        #print(self.input_matrix.shape)

        return (m, self.input_matrix[:, :(1 + idx), :],
                durations, job_reqs, (job_id_to_index, job_index_to_id))

    def get_action(self, state, argmax=False):

        m, input_matrix, durations, job_reqs, (id_to_index, index_to_id) = self.get_input(state)
        #print(m.shape)
        _state = torch.as_tensor(m, dtype=torch.float32)
        _input = torch.as_tensor(input_matrix,
            dtype=torch.float32)
        #print(_input.shape)
        enc, (h, c) = self.encode(_state, _input)

        batch_size = enc.shape[0]
        input_seq_len = enc.shape[1]
        aspace = np.arange(input_seq_len)
        stop_index = input_seq_len - 1

        chosen = -1
        loss = 1.0
        chosen_items = []
        cannot_placed = []
        ents = []
        #print("state shape",_state.shape)
        while chosen != stop_index:
            #print("!?")
            pointer_dist, (h, c) = self.decode_single(_state, enc, h, c)
            pointer_dist[0, stop_index] = -1e3
            for i in range(input_seq_len):
                if (i in chosen_items) or (i in cannot_placed):
                    pointer_dist[0, i] = -1e4
            #if torch.max(pointer_dist[0]) <= -1e5:
            #    break
            #print(pointer_dist[0])

            softmaxed = pointer_dist.softmax(1)
            #print(softmaxed[0].data.numpy())
            ents.append(get_entorpy(softmaxed.clone().data.numpy()))
            #print(softmaxed.data.numpy())
            if argmax is False:
                chosen = np.random.choice(aspace, p=softmaxed[0].data.numpy())
            else:
                chosen = np.argmax(softmaxed[0].data.numpy())
            ns = _state.clone()

            if chosen == stop_index:
                break
            else:
                ns[:durations[chosen]] -= job_reqs[chosen]
                if ns[:durations[chosen]].min() > 0:
                    _state = ns
                    chosen_items.append(chosen)
                else:
                    cannot_placed.append(chosen)

        return [index_to_id[idx] for idx in chosen_items], chosen_items, np.mean(ents)

    def get_action_(self, _state, _input, durations, job_reqs, id_to_index, index_to_id,
            argmax=False):

        enc, (h, c) = self.encode(_state, _input)

        input_seq_len = enc.shape[1]
        aspace = np.arange(input_seq_len)
        stop_index = input_seq_len - 1

        chosen = -1
        chosen_items = []
        cannot_placed = []
        ents = []
        #print("state shape",_state.shape)
        while chosen != stop_index:
            #print("!?")
            pointer_dist, (h, c) = self.decode_single(_state, enc, h, c)
            pointer_dist[0, stop_index] = -1e3
            for i in range(input_seq_len):
                if (i in chosen_items) or (i in cannot_placed):
                    pointer_dist[0, i] = -1e4
            #if torch.max(pointer_dist[0]) <= -1e5:
            #    break
            #print(pointer_dist[0])

            softmaxed = pointer_dist.softmax(1)
            #print(softmaxed[0].data.numpy())
            ents.append(get_entorpy(softmaxed.clone().data.numpy()))
            #print(softmaxed.data.numpy())
            if argmax is False:
                chosen = np.random.choice(aspace, p=softmaxed[0].data.numpy())
            else:
                chosen = np.argmax(softmaxed[0].data.numpy())
            ns = _state.clone()

            if chosen == stop_index:
                break
            else:
                ns[:durations[chosen]] -= job_reqs[chosen]
                if ns[:durations[chosen]].min() > 0:
                    _state = ns
                    chosen_items.append(chosen)
                else:
                    cannot_placed.append(chosen)

        return chosen_items, np.mean(ents)

    def get_s(self, _state, _input, durations):
        enc, (h, c) = self.encode(_state, _input)

        def input_tr(states):
            #print(states.shape)
            m_holder = torch.zeros(size=[1, 20, 3], dtype=torch.float32)

            m_holder.data[0, :, :] = states

            v = m_holder.view(m_holder.shape[0], -1)[:1]
            #print(v.shape)
            v.unsqueeze(1).shape
            return v.unsqueeze(1)
        state_input = self.state_transform(input_tr(_state))

        out, (h, c) = self.decoder(state_input, (h, c))
        out = self.nn2(F.relu(self.nn1(out[0])))
        return out

    def get_log_p_a_s(self, state, enc, durations, job_reqs, h, c, item_indices):

        chosen_items = []
        cannot_placed = []
        loss = 1.0
        input_seq_len = enc.shape[1]
        stop_index = input_seq_len - 1
        entropy = 0
        ent = 0
        item_indices.append(stop_index)
        for chosen in item_indices:
            pointer_dist, (h, c) = self.decode_single(state, enc, h, c)
            for i in range(input_seq_len):
                if i in chosen_items:
                    #already chosen items
                    pointer_dist[0, i] = -1e3
            softmaxed = pointer_dist.softmax(1)[0, :]
            _softmaxed = softmaxed + 0.0001
            ent = -torch.sum(_softmaxed * torch.log(_softmaxed))
            entropy += ent
            if chosen == stop_index:
                break
            loss *= softmaxed[chosen]
            state[:durations[chosen]] -= job_reqs[chosen]
            chosen_items.append(chosen)

        entropy = entropy
        loss *= softmaxed[stop_index]
        p_a_s = torch.log(loss)
        return p_a_s, entropy

    def train_single(self, state, action, adv):
        chosen_items = []
        cannot_placed = []
        loss = 1.0
        _action = copy.copy(action)
        m, input_matrix, durations, job_reqs, _ = self.get_input(state)
        _state = Variable(torch.as_tensor(m, dtype=torch.float32))
        _input = Variable(torch.as_tensor(input_matrix,
            dtype=torch.float32))
        enc, (h, c) = self.encode(_state, _input)
        #batch_size = enc.shape[0]
        input_seq_len = enc.shape[1]
        stop_index = input_seq_len - 1
        #_action.append(stop_index)
        for chosen in _action:
            pointer_dist, (h, c) = self.decode_single(_state, enc, h, c)
            for i in range(input_seq_len):
                if (i in chosen_items) or (i in cannot_placed):
                    pointer_dist[0, i] = -1e3
            softmaxed = pointer_dist.softmax(1)
            if chosen == stop_index:
                #loss *= softmaxed[0, stop_index]
                break

            _state[:durations[chosen]] -= job_reqs[chosen]
            chosen_items.append(chosen)
            loss *= softmaxed[0, chosen]
        #loss *= softmaxed[0, stop_index]
        loss = -(adv * torch.log(loss))
        return loss

    def decode_single(self, states, enc, h, c):
        batch_size = states.shape[0]
        def input_tr(states):
            #print(states.shape)
            m_holder = torch.zeros(size=[batch_size, 20, 3], dtype=torch.float32)

            m_holder.data[0, :, :] = states

            v = m_holder.view(m_holder.shape[0], -1)[:1]
            #print(v.shape)
            v.unsqueeze(1).shape
            return v.unsqueeze(1)
        state_input = self.state_transform(input_tr(states))
        #state_input = input_tr(states)
        #print("state_input_shape", state_input.shape)
        #print("stat#e_input.shape", state_input.shape)
        _, (h, c) = self.decoder(state_input, (h, c))
        query = h.squeeze(0)
        #assume batch_size to be 1

        pointer_dist = self.pointer(enc, query)
        return pointer_dist, (h, c)

    def decode_step(self, states, enc, h, c,):
        pass

    def decode(self, states, enc, h, c, durations, job_reqs):
        batch_size = states.shape[0]
        _states = states.clone()
        def input_tr(states):
            m_holder = torch.zeros(size=[batch_size, 20, 2], dtype=torch.float32)
            m_holder.data[0, :, :] = states
            v = m_holder.view(m_holder.shape[0], -1)[:1]
            v.unsqueeze(1).shape
            return v.unsqueeze(1)
        n_input_tasks = len(durations)
        #print(n_input_tasks)

        p_to_chosen = []
        chosen_list = []
        p = 1.0
        _chosen = 0
        while (_chosen != -1):
            _state_input = self.state_transform(input_tr(_states))
            _, (h, c) = self.decoder(_state_input, (h, c))
            query = h.squeeze(0)
            pointer_dist = self.pointer(enc, query)
            #for j in range(n_input_tasks):
            #    if j not in lrdict:
            #        pointer_dist[0, j] = -1e6
            #pointer_dist[0, n_input_tasks] = -1e3

            softmaxed = pointer_dist.softmax(1)
            #print(softmaxed)
            search_order = (-softmaxed).sort()[1].numpy()[0]
            #print(softmaxed)
            _chosen = -1
            idx = 0
            for chosen in search_order:
                if chosen == n_input_tasks:
                    # halt choosing jobs
                    break

                if chosen not in durations:
                    continue

                new_states = _states.clone()
                new_states[:durations[chosen]] -= job_reqs[chosen]
                if new_states[:durations[chosen]].min() > 0:
                    #print(new_states)
                    #machine = _m.view(-1)
                    del durations[chosen]
                    del job_reqs[chosen]
                    _states = new_states
                    _chosen = chosen
                    chosen_list.append(_chosen)
                    p_to_chosen.append(softmaxed[0, _chosen])
                    #print(chosen)
                    p *= softmaxed[0, _chosen]
                    break

            if _chosen == -1:
                p *= softmaxed[0, n_input_tasks]
        loss = -torch.log(p)
        return loss, p_to_chosen, chosen_list

    def forward(self, state, target_actions):
        pass