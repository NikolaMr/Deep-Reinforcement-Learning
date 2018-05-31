import numpy as np
import random

class ExplorationStrategy:
    def __init__(self):
        return
    def choose_action(self, agent):
        raise NotImplementedError
    def step(self):
        return

class EpsilonGreedyExplorationStrategy(ExplorationStrategy):
    def __init__(self, start_epsilon, end_epsilon, exploring_frames, num_actions):
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.current_epsilon = self.start_epsilon
        self.exploring_frames = exploring_frames
        self.num_actions = num_actions
    def step(self):
        if self.current_epsilon > self.end_epsilon:
            self.current_epsilon -= (self.start_epsilon - self.end_epsilon) / self.exploring_frames
    def choose_action(self, agent, s_t):
        #Choose an action by greedily (with e chance of random action) from the Q-network
        if np.random.rand(1) < self.current_epsilon:
            return random.randrange(self.num_actions)
        else:
            q = agent.model.predict(s_t)
            policy_max_Q = np.argmax(q)
            return policy_max_Q

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

class BoltzmannExplorationStrategy(ExplorationStrategy):
    def __init__(self, num_actions):
        self.num_actions = num_actions
    def choose_action(self, agent, s_t):
        q = agent.model.predict(s_t)
        q_soft = softmax(q[0])
        return np.random.choice(self.num_actions, p=q_soft)
