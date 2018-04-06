
import DuelingDDQNAgent

from SumTree import SumTree
import numpy as np

class PER_Memory():
    def calculate_priority(self, error):
        return (error + self.eps) ** self.alfa
    def __init__(self, buff_sz, eps, alfa):
        self.M = SumTree(buff_sz)
        self.eps = eps
        self.alfa = alfa
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
        return samples, indices
    def update(self, indices_with_priorities):
        for idx_p in indices_with_priorities:
            self.M.update(idx_p[0], idx_p[1])

class Dueling_DDQN_PER_Agent(DuelingDDQNAgent.Dueling_DDQN_Agent):
    def __init__(self, setup_dict=None):
        super(Dueling_DDQN_PER_Agent, self).__init__(setup_dict)
        if setup_dict == None:
            setup_dict = {}
        if 'alfa' not in setup_dict:
            setup_dict['alfa'] = 1.4
        if 'err_eps' not in setup_dict:
            setup_dict['err_eps'] = 1e-3
        self.alfa = setup_dict['alfa']
        self.err_eps = setup_dict['err_eps']

    def get_batch(self):
        self.sample_with_indices = self.memory.sample(self.replay_batch_size)
        return self.sample_with_indices[0]

    def initialize(self):
        self.memory = PER_Memory(self.replay_memory_size, self.err_eps, self.alfa)
        self.build_model()
        self.var_assign_ops = DuelingDDQNAgent.update_target_graph(self.target_model, self.model, self.tau)

    def save_transition(self, transition):
        #s_t, a_t, r_t_clipped, s_t1, done
        state_t = transition[0]
        state_t1 = transition[3]
        action_t = transition[1]
        reward_t = transition[2]
        done_t = transition[4]
        targets = self.get_target(state_t, action_t, reward_t, state_t1, done_t)

        q = self.model.predict(state_t)
        policy_max_Q = np.argmax(q)
        a_t = policy_max_Q
        error = abs(q[0][a_t] - np.max(targets))
        self.memory.append(transition, error)
    def post_train(self):
        super(Dueling_DDQN_PER_Agent, self).post_train()
        indices_with_priorities = []
        for i in range(len(self.sample_with_indices[0])): 
            sample = self.sample_with_indices[0][i]
            
            state_t = sample[0]
            action_t = sample[1]
            reward_t = sample[2]
            state_t1 = sample[3]
            done_t = sample[4]

            targets = self.get_target(state_t, action_t, reward_t, state_t1, done_t)

            q = self.model.predict(state_t)
            policy_max_Q = np.argmax(q)
            a_t = policy_max_Q
            error = abs(q[0][a_t] - np.max(targets))

            indices_with_priorities.append((self.sample_with_indices[1][i], self.memory.calculate_priority(error)))

            if reward_t != 0.0:
                self.cnt_rewarded += 1
        self.memory.update(indices_with_priorities)