from pdb import set_trace as T
import time
import numpy as np
from typing import Tuple
from random import randint
import matplotlib.pyplot as plt
# Use for .py file
from tqdm import tqdm
# Use for notebook
# from tqdm.notebook import tqdm

import gym
from gym import envs
import gym_pcgrl
# Use print to confirm access to local pcgrl gym
# print([env.id for env in envs.registry.all() if "gym_pcgrl" in env.entry_point])

import torch
from torch import nn
from torch import ByteTensor, Tensor
from torch.nn import Conv2d, CrossEntropyLoss

from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap


"""
/// Required Environment ///
conda create -n ribs-pt python=3.7
conda install -c conda-forge notebook
conda install pytorch torchvision torchaudio -c pytorch
pip install 'ribs[all]' gym~=0.17.0 Box2D~=2.3.10 tqdm
git clone https://github.com/amidos2006/gym-pcgrl.git
cd gym-pcgrl  # Must run in project root folder for access to pcgrl modules
"""

"""
/// Resources ///

Sam's example code:
https://github.com/smearle/gol-cmame/blob/master/gol_cmame.py

PCGRL Repo:
https://github.com/amidos2006/gym-pcgrl

Neural CA Paper:
https://arxiv.org/pdf/2009.01398.pdf

RIBS examples:
https://docs.pyribs.org/en/stable/tutorials/lunar_lander.html
"""

"""
Set Parameters
"""
env = gym.make("binary-wide-v0")
seed = 1339

def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)

#   if CUDA:
#       m.cuda()
#       m.to('cuda:0')

class NNGoL(nn.Module):
  def __init__(self):
    super().__init__()
    self.m = 5
    self.l1 = Conv2d(1, 2 * self.m, 3, 1, 1, bias=True)
    self.l2 = Conv2d(2 * self.m, self.m, 1, 1, 0, bias=True)
    self.l3 = Conv2d(self.m, 2, 1, 1, 0, bias=True)
    self.layers = [self.l1, self.l2, self.l3]
    self.apply(init_weights)

  def forward(self, x):
    with torch.no_grad():
        x = self.l1(x)
        x = torch.nn.functional.relu(x)
        x = self.l2(x)
        x = torch.nn.functional.relu(x)
        x = self.l3(x)
        x = torch.sigmoid(x)
        x = torch.stack([unravel_index(x[i].argmax(),x[i].shape) for i in range(x.shape[0])])
    # axis 0 is batch
    # axis 0,0 is the 0 or 1 tile
    # axis 0,1 is the x value
    # axis 0,2 is the y value 
    
    return x

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

    if type(m) == torch.nn.Conv2d:
        torch.nn.init.orthogonal_(m.weight)

def set_nograd(nn):
    for param in nn.parameters():
        param.requires_grad = False


def get_init_weights(nn):
  """
  Use to get dimension of weights from PyTorch
  """
  init_weights = []
  for lyr in nn.layers:
    init_weights.append(lyr.weight.view(-1).numpy())
    init_weights.append(lyr.bias.view(-1).numpy())
  init_weights = np.hstack(init_weights)

  return init_weights

def set_weights(model, weights):
  """
  Use to set model weights.
  """
  with torch.no_grad():
    n_a = 0  # start of slice
    n_b = 0  # end of slice

    # Convert to tensor
    weights = torch.tensor(weights).float()

    # Set l1 weights
    l1_len = len(model.l1.weight.flatten())
    n_b = l1_len
    l1_shape = model.l1.weight.shape
    l1_w = nn.Parameter(weights[n_a:n_b].reshape(l1_shape))
    model.l1.weight = l1_w

    # Set l1 Bias
    l1_b_len = len(model.l1.bias.flatten())
    n_a = n_b
    n_b += l1_b_len 
    l1_b_shape = model.l1.bias.shape
    l1_b = nn.Parameter(weights[n_a:n_b].reshape(l1_b_shape))
    model.l1.bias = l1_b

    # Set l2 weights
    l2_len = len(model.l2.weight.flatten())
    n_a = n_b
    n_b += l2_len
    l2_shape = model.l2.weight.shape
    l2_w = nn.Parameter(weights[n_a:n_b].reshape(l2_shape))
    model.l2.weight = l2_w

    # Set l2 Bias
    l2_b_len = len(model.l2.bias.flatten())
    n_a = n_b
    n_b += l2_b_len
    l2_b_shape = model.l2.bias.shape
    l2_b = nn.Parameter(weights[n_a:n_b].reshape(l2_b_shape))
    model.l2.bias = l2_b

    # Set l3 weights
    l3_len = len(model.l3.weight.flatten())
    n_a = n_b
    n_b += l3_len
    l3_shape = model.l3.weight.shape
    l3_w = nn.Parameter(weights[n_a:n_b].reshape(l3_shape))
    model.l3.weight = l3_w

    # Set l3 Bias
    l3_b_len = len(model.l3.bias.flatten())
    n_a = n_b
    n_b += l3_b_len
    l3_b_shape = model.l3.bias.shape
    l3_b = nn.Parameter(weights[n_a:].reshape(l3_b_shape))
    model.l3.bias = l3_b

    return model


def simulate(env, model, init_state, seed=None):
    """
    Function to run a single trajectory and return results.

    Args:
        env (gym.Env): A copy of the binary-wide-v0 environment.
        model (np.ndarray): The array of weights for the policy.
        seed (int): The seed for the environment.
    Returns:
        total_reward (float): The reward accrued by the lander throughout its
            trajectory.
        path_length (float): The path length of the final solution.
        regions (float): The number of distinct regions of the final solution.
    """
    if seed is not None:
        env.seed(seed)
    
    total_reward = 0.0
    path_length = 0.0
    regions = 0.0
    # Allow us to manually set the level-map on reset (using the "_old_map" attribute)
    env._rep._random_start = False
    env._rep._old_map = init_state
    obs = env.reset()
    done = False
    
    n_step = 0
    last_action = None
    while not done:
        # action = env.action_space.sample()
        in_tensor = torch.unsqueeze(torch.unsqueeze(torch.tensor(np.float32(obs['map'])),0), 0)
        action = model(in_tensor)[0].numpy()
        action = np.array([action[1],action[2],action[0]])
        # two identical actions means that we are caught in a loop, assuming we sample actions deterministically from the NN
        done = (action == last_action).all() or n_step >= 100
        if done:
            path_length = env._rep_stats['path-length']
            regions = env._rep_stats['regions']
            break
        last_action = action
        obs, reward, _, info = env.step(action)
        env.render()
        total_reward += reward
        n_step += 1
    
    return total_reward, path_length, regions


from ribs.archives import GridArchive

archive = GridArchive(
    [100,100],  # 10 bins in each dimension.
    [(1.0, 196.0),(1.0, 196.0)],  # path length and num rooms
)

model = NNGoL()
set_nograd(model)
initial_w = get_init_weights(model)
emitters = [
    ImprovementEmitter(
        archive,
        initial_w.flatten(),
        0.1,  # Initial step size.
        batch_size=30,
    ) for _ in range(5)  # Create 5 separate emitters.
]

optimizer = Optimizer(archive, emitters)

# This is the initial map which will act as a seed to our NCAs
init_state = np.random.randint(0, 2, (14, 14))
#init_state = np.zeros((14, 14))
#init_state[5:-5, 5:-5] = 1

start_time = time.time()
total_itrs = 1000
# total_itrs = 500

for itr in tqdm(range(1, total_itrs + 1)):
    # Request models from the optimizer.
    sols = optimizer.ask()

    # Evaluate the models and record the objectives and BCs.
    objs, bcs = [], []
    for model_w in sols:
        set_weights(model, model_w)
        obj, path_length, regions = simulate(env, model, init_state, seed)
        objs.append(obj)
        bcs.append([path_length, regions])

    # Send the results back to the optimizer.
    optimizer.tell(objs, bcs)

    # Logging.
    if itr % 1 == 0:
        df = archive.as_pandas(include_solutions=False)
        elapsed_time = time.time() - start_time
        print(f"> {itr} itrs completed after {elapsed_time:.2f} s")
        print(f"  - Archive Size: {len(df)}")
        print(f"  - Max Score: {df['objective'].max()}")

# # Visualize Result
# plt.figure(figsize=(8, 6))
# grid_archive_heatmap(archive, vmin=-300, vmax=300)
# plt.gca().invert_yaxis()  # Makes more sense if larger velocities are on top.
# plt.ylabel("Impact y-velocity")
# plt.xlabel("Impact x-position")

# Print table of results
df = archive.as_pandas()
# high_performing = df[df["objective"] > 200].sort_values("objective", ascending=False)
print(df)
