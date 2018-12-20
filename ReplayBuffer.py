import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory_buffer = []
        self.oldest_idx = 0


    def __len__(self):
        return len(self.memory_buffer)

    def add_memory(self, memory):
        if len(self.memory_buffer) < self.capacity:
            self.memory_buffer.append(memory)
        else:
            self.memory_buffer[self.oldest_idx] = memory
            self.oldest_idx = (self.oldest_idx + 1) % self.capacity


    def get_memory_batch(self, batch_size):
        idxs = np.random.choice(len(self.memory_buffer), batch_size).astype('int32')
        batch = [self.memory_buffer[i] for i in idxs]

        return batch


    def get_random_memory(self):
        idx = np.random.choice(len(self.memory_buffer))
        return self.memory_buffer[idx]