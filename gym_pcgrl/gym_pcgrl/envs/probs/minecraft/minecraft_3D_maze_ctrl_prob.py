import numpy as np

from gym_pcgrl.envs.probs.minecraft.minecraft_3D_maze_prob import Minecraft3DmazeProblem

"""
Generate a fully connected top down layout where the longest path is greater than a certain threshold
"""
class Minecraft3DmazeCtrlProblem(Minecraft3DmazeProblem):
    # We do these things in the ParamRew wrapper (note that max change and iterations
    pass
