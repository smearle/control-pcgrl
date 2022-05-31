import numpy as np

from gym_pcgrl.envs.probs.minecraft.minecraft_3D_Dungeon_prob import Minecraft3DDungeonProblem

"""
Generate a fully connected top down layout where the longest path is greater than a certain threshold
"""
class Minecraft3DDungeonCtrlProblem(Minecraft3DDungeonProblem):
    # NOTE: We do these things in the ParamRew wrapper.
    def get_episode_over(self, new_stats, old_stats):
        """ If the generator has reached its targets. (change percentage and max iterations handled in pcgrl_env)"""
        return False

    def get_reward(self, new_stats, old_stats):
        return None
