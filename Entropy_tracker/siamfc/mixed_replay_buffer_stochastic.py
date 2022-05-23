import random
import numpy as np
import os
import pickle
from collections import deque
import random

class mixed_replay_buffer_stochastic:
    def __init__(self, capacity,seed,tau=0.1):
        self.capacity=capacity
        self.buffer=[]
        self.hbuffer=[]
        self.tau=tau
        self.position1=0
        self.position2=0
        random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position1] = (state, action, reward, next_state, done)
        self.position1 = (self.position1 + 1) % self.capacity

    def hpush(self,state,action,reward,next_state,done):
        if len(self.hbuffer) < self.capacity:
            self.hbuffer.append(None)
        self.hbuffer[self.position2] = (state, action, reward, next_state, done)
        self.position2 = (self.position2 + 1) % self.capacity

    def sample(self, batch_size):
        batch = list()
        for i in range(batch_size):
            if len(self.hbuffer)<batch_size:
                batch=random.sample(self.buffer,batch_size)
                break
            else:
                r = random.uniform(0, 1)
                if r>self.tau:
                    transition=random.sample(self.buffer,1)

                else:
                    transition=random.sample(self.hbuffer,1)

                batch.append(*transition)
        state,action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity