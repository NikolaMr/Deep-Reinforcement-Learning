import numpy as np

class SumTree:
    
    def __init__(self, capacity):
        self.write = 0
        self.capacity = capacity
        self.tree = np.zeros(2 * self.capacity - 1)
        self.data = np.zeros(self.capacity, dtype=object)
        
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    def _retreive(self, idx, s):
        # since we count from 0
        left = 2 * idx + 1
        right = 2 * idx + 2
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retreive(left, s)
        else:
            return self._retreive(right, s - self.tree[left])
            
    def total(self):
        return self.tree[0]
        
    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        self.write %= self.capacity
        
    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
        
    def get(self, s):
        idx = self._retreive(0, s)
        dataIdx = idx - self.capacity + 1
        return self.data[dataIdx]
        
