from gym.envs.classic_control import rendering

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



class ClusteringEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n_resources=2, n_resource_slot_capacities=(20, 15), p_job_arrival=0.5, max_job_log_size=100,
                 max_job_slot_size=5, backlog_size=100, discount=1.0, time_horizon=5):

        self.max_job_log_size = max_job_log_size
        self.repre = 'image'
        self.backlog_size = backlog_size
        self.n_resources = n_resources
        self.p_job_arrival = p_job_arrival
        self.max_job_slot_size = max_job_slot_size
        self.discount = discount
        self.time_horizon = time_horizon
        self.n_resource_slot_capacities = n_resource_slot_capacities
        self.dist = dist(episode_max_size=1024, new_job_rate=p_job_arrival)
        self.prev_timestep_proceeded = False
        self.seq_num = 0
        self.seq_idx = 0
        self.renderer = None


        """
            MACHINE,
            JOB,
            JOB SLOT,
            JOB BACK LOG,
            JOB RECORD,
            MACHINE,
            EXTRA_INFO
        """
        self.state_space = spaces.Discrete(self.max_job_slot_size)
        self.action_space = spaces.Discrete(self.max_job_slot_size + 1) # num # n_resources denotes do nothing.
        self.scenario = self.dist.generate_work_sequence()

    def render(self, mode='human'):
        """
        fig = plt.figure("screen", figsize=(5, 5))

        if self.renderer is None:
            self.renderer = plt.figure("screen", figsize=(5, 5))
            self.renderer = rendering.SimpleImageViewer()


        for i in range(self.n_resources):
            plt.subplot(self.n_resources, 1, 1)
            plt.imshow(self.machine.graphical_view[0])

        canvas = FigureCanvas(fig)
        canvas.draw()       # draw the canvas, cache the renderer
        x, y, w, h  = fig.bbox.bounds
        w, h = int(w), int(h)
        img = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        print(img)
        print(img.shape)
        self.renderer.imshow(img)
        print(w)
        """

        a = "="

        title = "available"
        for i in range(self.max_job_slot_size):
            title += "\t\tjob slot %d" % (i+1)
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
                    x[1] += str(int(r))+","
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

        print("We're in timestep %d, timestamp %d" % (self.current_timestep, self.current_timestamp))
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
        print(b)





        pass

    def step(self, a):
        self._do_action(a)

        self.current_timestamp += 1
        done = self._is_finished()
        observation = self._observe()
        reward = self._get_reward()

        return observation, reward, done, False

    def reset(self, dist=None):
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = None
        self.scenario = self.dist.generate_work_sequence()
        self.last_timestamp_timestep = 0
        self.current_timestep = 0
        self.current_timestamp = 0
        self.job_slot = JobSlot(self.max_job_slot_size)
        self.job_backlog = JobBacklog(self.backlog_size)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo()
        self.machine = Machine(self.n_resources, self.time_horizon, self.n_resource_slot_capacities)


        if dist is not None:
            self.dist = dist


    def close(self):
        pass

    def seed(self, seed=None):
        if seed is None:
            seed = 1
        np.random.seed(seed)
        pass

    def _observe(self):

        machine_repr = self.machine.repr()
        job_slot_repr = self.job_slot.repr()
        return {
            "machine": machine_repr,
            "job_slot": job_slot_repr
        }


    def _do_action(self, a):
        self.last_timestamp_timestep = self.current_timestep
        self.sync()
        # can't pick job or slot is full;
        if (a == self.max_job_slot_size) or (self.job_slot[a] is None):
            self._handle_move()
        # pick job `a` and do
        else:
            self.handle_assign(a)
        if self.last_timestamp_timestep != self.current_timestep:
            self._proceed()


    def _proceed(self):
        # check there's finished jobs
        finished_job_info = self.machine.time_proceed(self.current_timestep)

    def sync(self):
        # check there is new job coming
        if self.scenario[self.current_timestep] is not None:
            #if there's an new job
            size = self.scenario[self.current_timestep]['size']
            duration = self.scenario[self.current_timestep]['duration']
            new_job = Job(size, duration, '?', self.current_timestep)
            self.job_backlog.append_job(new_job)

        if self.job_backlog.num_jobs() > 0:
            pass

        # move job in backlog to job slot if possible
        while True:
            if self.job_backlog.num_jobs() == 0:
                break
            if self.job_slot.num_empty_slots() == 0:
                break
            job = self.job_backlog.get_job()
            i = self.job_slot.assign(job)

    def _get_reward(self):

        # use negative number of unfinished jobs
        running_jobs = self.machine.get_num_unfinished_jobs()
        jobs_in_slots = self.job_slot.num_allocated_jobs()
        jobs_in_backlog = self.job_backlog.num_jobs()
        return -(running_jobs + jobs_in_slots + jobs_in_backlog)

        # is there any other metrics for use...

    def _handle_move(self):
        self.current_timestep += 1

    def handle_assign(self, i):
        job = self.job_slot[i]
        self.job_slot.release(i)
        self.machine.allocate_job(job, self.current_timestep)



    def _is_finished(self):
        return False
