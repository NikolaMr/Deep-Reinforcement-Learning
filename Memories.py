import numpy as np
import random
from SumTree import SumTree

class Memory():
    def __init__(self, buff_sz):
        self.buff_sz = buff_sz
    def save_experience(self, agent, transition):
        raise NotImplementedError
    def sample(self, num_samples):
        raise NotImplementedError
    def post_train(self, params):
        return

class ExperienceReplayMemory(Memory):
    def __init__(self, buff_sz):
        self.buff_sz = buff_sz
        self.M = np.zeros(buff_sz, dtype=object)
        self.write = 0
    def save_experience(self, agent, transition):
        self.append(transition)
    def append(self, tup):
        self.M[self.write] = tup
        self.write += 1
        self.write %= self.buff_sz
    def sample(self, num_samples):
        #minibatch = random.sample(self.M, num_samples)
        indices = np.random.randint(0,self.M.shape[0],num_samples)
        return self.M[indices]

class MemoryPrioritizedForgetting(Memory):
    def __init__(self, buff_sz):
        self.buff_sz = buff_sz
        self.M = np.zeros(buff_sz, dtype=object)
        self.write = 0
        self.elem_cnt = 0
    def save_experience(self, agent, transition):
        self.append(transition)
    def append(self, tup):
        self.M[self.write] = tup
        self.write += 1
        self.write %= self.buff_sz
        self.elem_cnt += 1
        if (self.elem_cnt > self.buff_sz):
            # register it has been filled once
            # in order to refill from now on
            self.elem_cnt = self.buff_sz
            dump = self.M[(self.write-1)%self.buff_sz]
            if dump[2] > 0.0:
                if random.random() < 0.7:
                    self.append(dump)
            elif dump[2] < 0.0:
                if random.random() < 0.4:
                    self.append(dump)
    def sample(self, num_samples):
        #minibatch = random.sample(self.M, num_samples)
        indices = np.random.randint(0,self.M.shape[0],num_samples)
        return self.M[indices]

class PrioritizedExperienceReplayMemory():
    def calculate_priority(self, error):
        return (error + self.eps) ** self.alfa
    def __init__(self, buff_sz, eps, alfa):
        self.M = SumTree(buff_sz)
        self.eps = eps
        self.alfa = alfa
    def save_experience(self, agent, transition):
        #s_t, a_t, r_t_clipped, s_t1, done
        state_t = transition[0]
        state_t1 = transition[3]
        action_t = transition[1]
        reward_t = transition[2]
        done_t = transition[4]
        targets = agent.get_target(state_t, action_t, reward_t, state_t1, done_t)

        q = agent.model.predict(state_t)
        policy_max_Q = np.argmax(q)
        a_t = policy_max_Q
        error = abs(q[0][a_t] - np.max(targets))
        self.append(transition, error)
    def append(self, tup, error):
        priority = self.calculate_priority(error)
        self.M.add(priority, tup)
    def sample(self, num_samples):
        total = self.M.total()
        s_arr = np.random.uniform(0.0, total, num_samples)
        samples = []
        indices = []
        for i in range(len(s_arr)):
            sample, idx = self.M.get(s_arr[i])
            samples.append(sample)
            indices.append(idx)
        # remember these in order to correct them in post train step
        self.last_samples = samples
        self.last_indices = indices
        return np.array(samples)
    def update(self, indices_with_priorities):
        for idx_p in indices_with_priorities:
            self.M.update(idx_p[0], idx_p[1])
    def post_train(self, agent):
        indices_with_priorities = []
        for i in range(len(self.last_samples)):
            sample = self.last_samples[i]

            state_t = sample[0]
            action_t = sample[1]
            reward_t = sample[2]
            state_t1 = sample[3]
            done_t = sample[4]

            targets = agent.get_target(state_t, action_t, reward_t, state_t1, done_t)

            q = agent.model.predict(state_t)
            policy_max_Q = np.argmax(q)
            a_t = policy_max_Q
            error = abs(q[0][a_t] - np.max(targets))

            indices_with_priorities.append((self.last_indices[i], self.calculate_priority(error)))

        self.update(indices_with_priorities)
