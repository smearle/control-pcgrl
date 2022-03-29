#!/usr/bin/env python3


########## IMPORTS ########### {{{1
import numpy as np
from gym.envs.box2d import BipedalWalker, BipedalWalkerHardcore
import warnings
#from timeit import default_timer as timer


########## BIPEDAL ########### {{{1

class QDBipedalWalker(BipedalWalker):

    def step(self, action):
        state, reward, done, o = super().step(action)

        """
        # hull
        self.hull.angle,        #0 # Normal angles up to 0.5 here, but sure more is possible.
        2.0*self.hull.angularVelocity/FPS, # 1
        0.3*vel.x*(VIEWPORT_W/SCALE)/FPS,  # 2 Normalized to get -1..1 range
        0.3*vel.y*(VIEWPORT_H/SCALE)/FPS,  # 3

        # leg0 j0
        # This will give 1.1 on high up, but it's still OK
        # (and there should be spikes on hiting the ground,
        # that's normal too)
        self.joints[0].angle, # 4
        self.joints[0].speed / SPEED_HIP, # 5

        # leg0 j1
        self.joints[1].angle + 1.0, # 6
        self.joints[1].speed / SPEED_KNEE, # 7

        # leg0 contact
        1.0 if self.legs[1].ground_contact else 0.0, # 8

        # leg1 j0
        self.joints[2].angle, # 9
        self.joints[2].speed / SPEED_HIP, # 10
        # leg1 j1
        self.joints[3].angle + 1.0, # 11
        self.joints[3].speed / SPEED_KNEE, # 12

        # leg1 contact
        1.0 if self.legs[3].ground_contact else 0.0 # 13
        """

        total_torque = 0
        for a in action:
            torque =  np.clip(np.abs(a), 0, 1)
            total_torque += torque

        features = np.array([
                    self.hull.position[0],      # 0 distance traveled
                    np.abs(state[0]),    # 1 head stability
                    np.abs(total_torque),# 2 torque per step
                    state[8] and state[13], # 3 legs up, jump 

                    np.abs(state[4]), # 4 leg0 hip angle
                    np.abs(state[5]), # 5 leg0 hip speed
                    np.abs(state[6]), # 6 leg0 knee angle
                    np.abs(state[7]), # 7 leg0 knee speed

                    np.abs(state[9]), # 8 leg1 hip angle
                    np.abs(state[10]),# 9 leg1 hip speed
                    np.abs(state[11]),# 10 leg1 knee angle
                    np.abs(state[12]) # 11 leg1 knee speed
                   ])
        return np.array(state), reward, features, done, o

class QDBipedalWalkerHardcore(QDBipedalWalker, BipedalWalkerHardcore):
    pass


########## Simulation functions ########### {{{1

def simulate(model, env, render_mode=False, num_episode=5):
    #reward_list = []
    #t_list = []
    max_episode_length = 3000
    episodes_reward_sum = 0
    episodes_feature_sum = (0,) * 12
    total_reward = 0.0
    total_features = (0,) * 12

    for episode in range(num_episode):
        #start_time = timer()
        obs = env.reset()

        if obs is None:
            obs = np.zeros(model.input_size)

        for t in range(max_episode_length):
            if render_mode:
                env.render("human")

            #action = model.get_action(obs, t=t, mean_mode=False)
            action = model.get_action(obs)
            prev_obs = obs
            obs, reward, features, done, info = env.step(action)

            if render_mode:
                pass
                #print("action", action, "step reward", reward)
                #print("step reward", reward)
            total_reward += reward
            total_features = tuple(x + y for x,y in zip(total_features, features))

            if done:
                break

        if render_mode:
            print("reward", total_reward, "timesteps", t)
        #reward_list.append(total_reward)
        #t_list.append(t)
        #duration = timer() - start_time
        #print(f"DEBUG simulate duration: {duration / t}")

    total_features = tuple(x + y for x,y in zip(total_features, features))
    total_features = tuple(x/t for x in total_features)

    episodes_reward_sum += total_reward
    episodes_feature_sum = tuple(x + y for x,y in zip(episodes_feature_sum, total_features))

    episode_avg_reward = episodes_reward_sum / num_episode
    episode_avg_features = tuple(x/num_episode for x in episodes_feature_sum)

    #return reward_list, t_list
    #print("MODEL: REWARD", episode_avg_reward)
    #print("MODEL: AVG FEATURES (orig, leg0, leg1)", episode_avg_features)
    #return tuple(reward_list), tuple(total_features)

    #scores = {'AvgReward': episode_avg_reward, 'Distance',

    scores = {
            "meanAvgReward": episode_avg_reward,
            "meanDistance": episode_avg_features[0],
            "meanHeadStability": episode_avg_features[1],
            "meanTorquePerStep": episode_avg_features[2],
            "meanJump": episode_avg_features[3],
            "meanLeg0HipAngle": episode_avg_features[4],
            "meanLeg0HipSpeed": episode_avg_features[5],
            "meanLeg0KneeAngle": episode_avg_features[6],
            "meanLeg0KneeSpeed": episode_avg_features[7],
            "meanLeg1HipAngle": episode_avg_features[8],
            "meanLeg1HipSpeed": episode_avg_features[9],
            "meanLeg1KneeAngle": episode_avg_features[10],
            "meanLeg1KneeSpeed": episode_avg_features[11]
    }

    #return (episode_avg_reward,), tuple(episode_avg_features)
    return scores


def make_env(env_name, seed=-1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if env_name.startswith("BipedalWalker"):
            if env_name.startswith("BipedalWalkerHardcore"):
                env = QDBipedalWalkerHardcore()
            else:
                env = QDBipedalWalker()

    if (seed >= 0):
        env.seed(seed)

    #print("environment details")
    #print("env.action_space", env.action_space)
    #print("high, low", env.action_space.high, env.action_space.low)
    #print("environment details")
    #print("env.observation_space", env.observation_space)
    #print("high, low", env.observation_space.high, env.observation_space.low)
    #assert False
    return env



class Model:
    ''' Simple MLP '''
    def __init__(self, config):
        self.layers = list(config["layers"])
        self.input_size = config["input_size"]
        self.output_size = config["output_size"]
        self.init_nn()

    def init_nn(self):
        layers_size = [self.input_size] + self.layers + [self.output_size]
        self.shapes = []
        for i in range(1, len(layers_size)):
            fst = layers_size[i-1]
            snd = layers_size[i]
            self.shapes.append((fst, snd))

        self.weight = []
        self.bias = []
        self.param_count = 0

        for shape in self.shapes:
            self.weight.append(np.zeros(shape=shape))
            self.bias.append(np.zeros(shape=shape[1]))
            self.param_count += (np.product(shape) + shape[1])

    def get_action(self, x):
        h = np.array(x).flatten()
        nb_layers = len(self.weight)
        for i in range(nb_layers):
            w = self.weight[i]
            b = self.bias[i]
            h = np.matmul(h, w) + b
            h = np.tanh(h)
        return h

    def set_model_params(self, model_params):
        pointer = 0
        for i in range(len(self.shapes)):
            w_shape = self.shapes[i]
            b_shape = self.shapes[i][1]
            s_w = np.product(w_shape)
            s = s_w + b_shape
            chunk = np.array(model_params[pointer:pointer+s])
            self.weight[i] = chunk[:s_w].reshape(w_shape)
            self.bias[i] = chunk[s_w:].reshape(b_shape)
            pointer += s


# MODELINE  "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
