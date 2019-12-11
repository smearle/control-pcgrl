import gym
import gym_pcgrl

env = gym.make('simcity-wide-v0')
obs = env.reset()
for t in range(10000000):
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
 #env.render()
  if done:
 #  print("Episode finished after {} timesteps".format(t+1))
    env.reset()
