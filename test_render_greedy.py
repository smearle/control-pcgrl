import time
import json
import os
import shutil
import PIL
import hydra
import cv2
from queue import PriorityQueue
import numpy as np
import copy
import random
from control_pcgrl.configs.config import Config
from control_pcgrl.rl.envs import make_env
from control_pcgrl.rl.utils import validate_config



class PrioritizedEnv:
    def __init__(self, priority1, priority2, env):
        self.priority1 = priority1
        self.priority2 = priority2
        self.env = env
    
    def __lt__(self, other):
        if self.priority1 == other.priority1:
            if self.priority2 == other.priority2:
                return random.random() < 0.5  # Randomly select one if both priorities are equal
            else:
                return self.priority2 < other.priority2
        else:
            return self.priority1 < other.priority1

def get_state_key(env):
    observation = env.unwrapped._rep.get_observation()
    map_key = key(observation['map'])
    pos_tuple = tuple(observation['pos'])
    return (map_key, pos_tuple)


def key(val):
    return tuple(map(tuple, val))


@hydra.main(config_path="control_pcgrl/configs", config_name="enjoy")
def test_render(cfg: Config):
    
    validate_config(cfg)
    env = make_env(cfg)
    NUM_EXPS = 1  # Define the number of experiments to run

    for exp in range(NUM_EXPS):
        trajectory = []
        done = False  
        obs = env.reset()
        trajectory.append(obs)
        queue = PriorityQueue()

        queue.put(PrioritizedEnv(-env.unwrapped._rep_stats["path-length"], -env.unwrapped._rep_stats["regions"], env))

        visited = []
        parent = {}
        actions = {}

        visited.append(get_state_key(env))
        parent[get_state_key(env)] = None
        actions[get_state_key(env)] = None
        print(f"Start Experiment {exp + 1}:")

        t = 30
        MAX_TIME = 60*t  # seconds
        INTERVAL = 10  # Print every 10 seconds
        RENDER = False
        num_nodes_explored = 0
        max_node = None
        max_length = -env.unwrapped._rep_stats["path-length"]
        start_time = time.time()
        start = True
        last_print_time = start_time

        while not done and not queue.empty():

            current_time = time.time()

            # Check if it's time to print the status
            if current_time - last_print_time >= INTERVAL or start:
                print(f"Time: {current_time - start_time:.2f} seconds elapsed")
                print(f"Nodes Explored:{num_nodes_explored}")
                print(f"Nodes in Frontier:{queue.qsize()}")
                print(f"Max Length Reached:{max_length}")
                print("-----------------------------------")
                last_print_time = current_time
                start = False

            if time.time() - start_time > MAX_TIME:  # Stop if more than MAX_TIME seconds have passed
                print("Time limit exceeded. Stopping.")
                break

            # print("Size:", queue.qsize())
            state = queue.get()
            # Get the key of the current state
            current_state_key = get_state_key(state.env)
            num_nodes_explored += 1

            for act in range(env.action_space.n):
                envCopy = copy.deepcopy(state.env)
                obs, rew, done, truncated, info = envCopy.step(act)
                if RENDER:
                    im = envCopy.render()
                    cv2.imshow('image', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                if get_state_key(envCopy) not in visited:
                    queue.put(PrioritizedEnv(-envCopy.unwrapped._rep_stats["path-length"], -envCopy.unwrapped._rep_stats["regions"], envCopy))
                    visited.append(get_state_key(envCopy))
                    child_state_key = get_state_key(envCopy)
                    parent[child_state_key] = current_state_key
                    actions[child_state_key] = act
                    # Check if the current state has a greater path length than the maximum seen so far
                    if envCopy.unwrapped._rep_stats["path-length"] > max_length:
                        max_length = envCopy.unwrapped._rep_stats["path-length"]
                        max_node = child_state_key
        if RENDER:  
            cv2.destroyAllWindows()

        # Start from the max_node
        current_node = max_node
        actions_list = []
        print("Maximum length found:", max_length)
        # Traverse the trajectory backward until reaching the start node
        while current_node != None:
            # Get the action that leads from the parent state to the current state
            # print(actions[current_node])
            action = actions[current_node]

            # Append the action to the actions list
            actions_list.append(action)

            # Get the parent state key from the parent dictionary
            parent_state_key = parent[current_node]
            # Move to the parent node in the trajectory
            current_node = parent_state_key

        # Reverse the actions list to get it from start to end
        actions_list.reverse()
        # print("Actions_List:", actions_list)
        actions_list = actions_list[1:]

        # Create a directory to store the trajectories if it doesn't exist
        trajectory_dir = "greedy_trajectories"
        if not os.path.exists(trajectory_dir):
            os.makedirs(trajectory_dir)

        exp_dir = os.path.join(trajectory_dir, f"trajectory_{exp}")
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
        os.makedirs(exp_dir)
            
        os.path.join(trajectory_dir, f"trajectory_{exp}.npy")
        for act in actions_list:
            obs, rew, done, truncated, info = env.step(act)
            trajectory.append(obs)
            im = env.render()
            # Convert the image from RGB to BGR (OpenCV uses BGR by default)
            im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.imshow('image', im_bgr)
            frame_filename = f'frame_{len(trajectory) - 1}.png'

            frame_directory = os.path.join(exp_dir, frame_filename)

            cv2.imwrite(frame_directory, im_bgr)
            cv2.waitKey(10)
        cv2.destroyAllWindows()
        # print(len(trajectory), len(actions_list))

        # Save the trajectory to a .npy file
        trajectory_file = os.path.join(trajectory_dir, f"trajectory_{exp}.npy")
        np.save(trajectory_file, trajectory)

if __name__ == '__main__':
    test_render()  
