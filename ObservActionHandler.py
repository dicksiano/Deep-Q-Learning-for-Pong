import gym
import collections
import numpy as np

# Constants
SKIP_ACT = 4
SKIP_OBS = 2

class ObservActionHandler(gym.Wrapper):
    def __init__(self, env):
        super(ObservActionHandler, self).__init__(env)
        self.observ_buffer = collections.queue(maxlen=SKIP_OBS)

def step(self, action):
    reward_total = 0.0
    done = False
    for _ in range(SKIP_ACT):
        if not done:
            observation, reward, done, info = self.env.step(action)
            self.observ_buffer.append(observation)
            reward_total = reward_total + reward
        
    real_observation = np.max( np.stack(self.observ_buffer), axis=0)
    return real_observation, reward_total, done, info

def reset(self):
    self.observ_buffer.clear()
    observation = self.env.reset()
    self.observ_buffer.append(observation)

    return observation