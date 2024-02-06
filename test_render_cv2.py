import time
import PIL
import hydra
import cv2

from control_pcgrl.configs.config import Config
from control_pcgrl.rl.envs import make_env
from control_pcgrl.rl.utils import validate_config


@hydra.main(config_path="control_pcgrl/configs", config_name="enjoy")
def test_render(cfg: Config):
    
    validate_config(cfg)
    env = make_env(cfg)

    while True:
        done = False
        obs = env.reset()
        # Randomly sample actions
        for i in range(512):
            obs, rew, done, truncated, info = env.step(env.action_space.sample())
            im = env.render()
            # Use OpenCV to display the image
            cv2.imshow('image', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)


if __name__ == '__main__':
    test_render()