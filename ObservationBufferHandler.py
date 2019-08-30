import gym
import numpy as np

STEPS = 4

class ObservationBufferHandler(gym.ObservationWrapper):
    def __init__(self, env, dtype=np.float32):
        super(ObservationBufferHandler, self).__init__(env)
        self.type = dtype

        previous_space = env.observation_space
        self.observation_space = gym.spaces.Box(
                                                previous_space.low.repeat(STEPS, axis=0), 
                                                previous_space.high.repeat(STEPS, axis=0), 
                                                dtype=dtype) 

    def observation(self, observation):
        self.buffer.append(observation)
        self.buffer = self.buffer[1:]

        return self.buffer
        
    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=dtype)
        return self.observation(self.env.reset())

    