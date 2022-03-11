from pdb import set_trace as TT

import gym
from mlg_wb_specs import MLGWB

# In order to use the environment as a gym you need to register it with gym
abs_MLG = MLGWB()
abs_MLG.register()
env = gym.make("MLGWB-v0")

# this line might take a couple minutes to run
obs  = env.reset()

# Renders the environment with the agent taking noops
done = False
while not done:
    env.render()
    # a dictionary of actions. Try indexing it and changing values.
    action = env.action_space.noop()
    obs, reward, done, info = env.step(action)