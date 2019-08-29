import gym

class EnvHandler(gym.Wrapper):
    def __init__(self, env):
        super(EnvHandler, self).__init__(env)

    def step(self, action):
        return self.env.step.action()

    def reset(self):
        self.env.reset()
        
        for i in [1,2]:
            observation, reward, done, info = self.env.step(i)
            if done:
                self.env.reset()
        return observation