# agent/experience_replay.py

import random
from collections import deque

class ExperienceReplay:
    def __init__(self, max_size=2000):
        self.memory = deque(maxlen=max_size)

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def size(self):
        return len(self.memory)
