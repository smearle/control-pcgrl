import logging
from pdb import set_trace as TT

import gym
import minerl

logging.basicConfig(level=logging.DEBUG)

env = gym.make('MineRLNavigateDense-v0')

done = False

obs = env.reset()

while not done:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    TT()
    env.render()