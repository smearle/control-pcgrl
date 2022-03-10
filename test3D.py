"""
Run random agent to test the 3D environment
"""
import gym
import gym_pcgrl
from utils import make_vec_envs

if __name__=="__main__":
    env = gym.make('minecraft_3D_maze-narrow3D-v0')
    while True:
        observation = env.reset()
        for step in range(500):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print(env._rep_stats)
            env.render()
