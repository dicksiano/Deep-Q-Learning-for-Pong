import gym
import numpy as np
import cv2

# Constants
IMAGE_SIZE = 84

REGULAR_SHAPE_1 = [210, 160, 3]
REGULAR_SIZE_1  = 210 * 160 * 3

REGULAR_SHAPE_2 = [250, 160, 3]
REGULAR_SIZE_2  = 250 * 160 * 3
 
DESIRED_SHAPE = [84,84,1]

class FrameInputHandler(gym.ObservationWrapper):
    def __init__(self, env):
        super(FrameInputHandler, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(IMAGE_SIZE,IMAGE_SIZE,1), dtype=np.uint8)

    def observation(self, obs):
        if obs.size == REGULAR_SIZE_1:
            img = np.reshape(obs, REGULAR_SHAPE_1).astype(np.float32)
        elif obs.size == REGULAR_SHAPE_2:
            img = np.reshape(obs, REGULAR_SHAPE_2).astype(np.float32)
        else:
            raise Exception("Unknow frame format!")
        
        img = self.rgbToGray(img)
        resized_img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)

        input_processed = resized_img[18:102, :]
        input_processed = np.reshape(input_processed, DESIRED_SHAPE)

        return input_processed.astype(np.uint8)

    # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    def rgbToGray(self, rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray