import gym 
import numpy as np

class PyTorchImageHandler(gym.ObservationWrapper):
    def __init__(self, env):
        super(PyTorchImageHandler, self).__init__(env)
        
        previous_shape = self.observation_space.shape                                            # HWC
        new_shape = (previous_shape[2],  previous_shape[0], previous_shape[1])                   # CHW
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=new_shape, dtype=np.float32)  # HWC -> CHW

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0) # HWC -> CHW