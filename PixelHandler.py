import gym
import numpy as np

PIXEL_RANGE = 255.0

class PixelHandler(gym.ObservationWrapper):
    def observation(self, observ):
        return np.array(observ).astype(np.float32) / PIXEL_RANGE