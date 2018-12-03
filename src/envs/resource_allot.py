from time import sleep
from .job_distribution import dist
from .variables import Job, JobBacklog, JobRecord, JobSlot, ExtraInfo, Machine
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import math
import heapq
import numpy as np

import gym
from gym import error
from gym import spaces
from gym import utils
from gym.utils import seeding

def flatten(m):
    state_dim = np.prod(m.shape)
    return np.reshape(m, newshape=(state_dim,))

class env():
    metadata = {'render.modes': ['human']}
    def __init__(self, episode_size=1000, force_stop=2000, observation_mode='image',
                n_resource_slot_capacities=(10, 10), p_job_arrival=0.3, max_job_length=15,
                num_slots=5, backlog_size=60, time_horizon=20):
        self.episode_size = episode_size
        self.force_stop = max(force_stop, 2 * self.episode_size)
        self.observation_mode = observation_mode
        self.repre = 'image'
        self.backlog_size = backlog_size
        self.n_resources = len(n_resource_slot_capacities)
        self.p_job_arrival = p_job_arrival
        self.num_slots = num_slots
        self.time_horizon = max(time_horizon, max_job_length)
        self.n_resource_slot_capacities = n_resource_slot_capacities
        self.max_job_length = max_job_length
        self.dist = dist(
            episode_max_size=self.episode_size, new_job_rate=p_job_arrival,
            job_len=max_job_length, num_resources=self.n_resources, p_small_job=0.9, max_resource_usage=30)
        self.seq_num = 0
        self.seq_idx = 0
        self.renderer = None

        self.state_space = spaces.Discrete(self.num_slots)
        self.action_space = spaces.Discrete(self.num_slots + 1) # num # n_resources denotes do nothing.
        self.scenarios = [self.dist.generate_work_sequence() for _ in range(200)]
        self.current_scenario = None

    def render(self, mode='human'):
        a = "="

        title = "available"
        for i in range(self.num_slots):
            title += "\t\tjob slot %d" % i
        txt = ["" for _ in range(self.time_horizon)]
        for i in range(self.time_horizon):
            for j in self.machine.avbl_slot[i]:
                txt[i] += str(int(j)) + ' '
        y = []

        for j, job in enumerate(self.job_slot.slot):


            x = ["NO JOB" for _ in range(6)]
            if job is not None:
                x[0] = "usage"
                x[1] = "("
                for r in job.res_vec:
                    x[1] += str(int(r)) + ","
                x[1] = str(x[1][:-1])
                x[1] += ")"
                x[2] = "length"
                x[3] = "  " + str(job.len)
                x[4] = "entered"
                x[5] = "time " + str(job.enter_time)
            for i in range(len(x)):
                x[i] = '\t\t\t' + x[i]
            y.append(x)
        b = a * 200
        print(b)

        print("We're in timestep %d" % (self.current_timestep))
        print(title)
        for i in range(max(self.time_horizon, 6)):
            if i < self.time_horizon:
                _t = txt[i]
            else:
                _t = ""
            for j in range(len(y)):
                if i < 6:
                    _t += y[j][i]
            print(_t)
        print("There are %d jobs waiting in job backlog" % self.job_backlog.num_jobs())
        print(b)

        pass

    def step(self, job_list):
        self.current_timestep += 1
        # check there is new job coming

        if self.episode_size <= self.current_timestep:
            pass
        else:
            #print("new job at this step ", len(self.current_scenario[self.current_timestep]))
            for job in self.current_scenario[self.current_timestep]:
                size = job['size']
                duration = job['duration']
                if duration != 0:
                    new_job = Job(size, duration, len(self.job_record.record), self.current_timestep)
                    try:
                        self.job_backlog.append_job(new_job)
                    except:
                        print("backlog is full!")
                        self.render()
                    self.extra_info.new_job_comes()
                    self.job_record.record[new_job.id] = new_job

        # move job in backlog to job slot if possible
        self._dequeue_backlog()

        for job in job_list:
            if self.job_slot.slot[job] is None:
                print("request jobs that are not in the slot")
                temp = [(x, y is not None) for (x, y) in enumerate(self.job_slot.slot)]
                print(job_list)
                print(temp)
                continue
            assigned = self.handle_assign(job)
            if assigned is True:
                #print("normal assign")
                pass
            else:

                print(job_list)
                #print("job length:", self.job_slot.slot[job].len)
                print("Not assigned job %d" % job)
                self.render()

                sleep(2.0)

                pass

        # move job in backlog to job slot if possible
        self._dequeue_backlog()

        self.extra_info.time_proceed()
        durations, taken_times = self.machine.time_proceed(self.current_timestep)

        done = self._is_finished()
        observation = self._observe()
        reward = self._get_reward()
        info = self.job_record.record

        return observation, reward, done, info

    def reset(self, dist=None, seq_no=0):
        if self.renderer is not None:
            self.renderer.close()

        if dist is not None:
            self.dist = dist

        if seq_no >= 0:
            self.current_scenario = self.scenarios[seq_no]

        self.renderer = None
        self.current_timestep = 0
        self.job_slot = JobSlot(self.num_slots)
        self.job_backlog = JobBacklog(self.backlog_size)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo()
        self.machine = Machine(
            self.max_job_length, self.n_resources,
            self.time_horizon, self.n_resource_slot_capacities)
        return self._observe()


    def close(self):
        pass

    def seed(self, seed=None):
        if seed is None:
            seed = 1
        np.random.seed(seed)
        pass

    def _observe(self, ob_as_dict=False):

        machine_repr = self.machine.repr()
        job_slot_repr = self.job_slot.repr()
        job_backlog_repr = self.job_backlog.repr()
        extra_info = self.extra_info.extra_info()

        observation = {
            "machine": machine_repr,
            "job_slot": job_slot_repr,
            "job_backlog": job_backlog_repr,
            "extra_info": extra_info
        }
        return observation

    def observation(self):
        return self._observe()

    def observation_to_rnn_sequence(self, one_hot=False):
        machine = self.machine.repr()
        time_horizon = machine.shape[0]
        rep = []
        for i, cap in enumerate(self.n_resource_slot_capacities):
            rep_one = None
            if one_hot is True:
                rep_one = np.zeros(shape=(time_horizon, cap + 1))
            else:
                rep_one = np.zeros(shape=(time_horizon, cap))
            rep.append(rep_one)
            for t in range(time_horizon):
                if one_hot is False:
                    rep[i][t, :machine[t][i]] = 1
                else:
                    rep[i][t, machine[t][i]] = 1
        job_reprs = []

        for job in self.job_slot.slot:
            if job is None:
                job_reprs.append(None)
                #print("nojob!", job_repr.shape)
                continue

            reprs = []
            for i, cap in enumerate(self.n_resource_slot_capacities):
                res_one = None
                if one_hot is True:
                    res_one = np.zeros(shape=(cap + 1,))
                    res_one[job.res_vec[i]] = 1
                else:
                    res_one = np.zeros(shape=(cap,))
                    res_one[job.res_vec[i]:] = 1
                reprs.append(res_one)
            reprs.append([self.current_timestep - job.enter_time])
            job_repr = np.hstack(reprs)
            #print("job!", job_repr.shape)
            job_reprs.append(job_repr)
        #print(job_reprs)
        #job_reprs = np.vstack(job_reprs)
        machine_repr = np.hstack(rep)
        return flatten(machine_repr), job_reprs

    def _dequeue_backlog(self):
        while True:
            if self.job_backlog.num_jobs() == 0:
                return
            if self.job_slot.num_empty_slots() == 0:
                return
            job = self.job_backlog.get_job()
            self.job_slot.assign(job)

    def _get_reward(self):
        reward = 0
        #print("--")

        #reward = -np.sum(self.machine.avbl_slot[0] / np.array([100, 100, 100]))
        #return reward
        #print("--")

        for job in self.machine.running_jobs:
            if job is None:
                continue
            reward += -1.0 / float(job.len)

        for job in self.job_slot.slot:
            if job is None:
                continue
            reward += -1.0 / float(job.len)


        for job in self.job_backlog.backlog:
            if job is None:
                continue
            reward += -1.0 / float(job.len)
        return reward



    def _handle_move(self):
        self.current_timestep += 1

    def handle_assign(self, i):
        job = self.job_slot[i]
        if job is None:
            return False
        if self.machine.allocate_job(job, self.current_timestep) is True:
            self.job_record.record[job.id] = job
            self.job_slot.release(i)
            return True
        else:
            return False

        self._dequeue_backlog()



    def _is_finished(self):
        ret = self.force_stop <= self.current_timestep
        ret2 = self.current_timestep >= self.episode_size
        ret2 &= self.job_slot.num_allocated_jobs() == 0
        ret2 &= self.job_backlog.num_jobs() == 0
        ret2 &= self.machine.get_num_unfinished_jobs() == 0
        return ret | ret2
