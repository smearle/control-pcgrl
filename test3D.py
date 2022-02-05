"""
Run random agent to test the 3D environment
"""
import imp
import gym
import gym_pcgrl
from utils import make_vec_envs

if __name__=="__main__":
    env = gym.make('minecraft_3D_maze-narrow3D-v0')
    observation = env.reset()
    for step in range(20):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render()
