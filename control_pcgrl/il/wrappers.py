import gym
import numpy as np
from control_pcgrl.envs.reps.narrow_rep import NarrowRepresentation

from control_pcgrl.envs.reps.wrappers import RepresentationWrapper


class PoDWrapper(gym.Wrapper):
    """Wrapper for generating random paths of destruction."""
    def __init__(self, env, cfg):
        self.env = env
        self.env.adjust_param(cfg=cfg)

        self.rep = self.unwrapped._rep
        self.prob = self.unwrapped._prob

        # Kind of a hack. Fine as long as we don't need to overwrite any of the init functionality in the representation
        # wrapper.
        self.rep = PoDRepresentationWrapper(rep=self.rep, cfg=cfg)

    def queue_goal_map(self, goal_map: np.ndarray):
        self.unwrapped._rep.unwrapped._old_map = goal_map
        self.unwrapped._rep.unwrapped._random_start = False

    def get_inverse_action_pair(self):
        destroy_action = self.action_space.sample()

        repair_action = self.get_repair_action()

        return destroy_action, repair_action 

    def get_repair_action(self):

        # FIXME: Specific to narrow. Move this to representation classes.
        curr_tile_int = self.rep.unwrapped._map[tuple(self.rep.unwrapped._pos)]

        # Account no-op in narrow representation.
        # curr_tile_narrow_build_action = curr_tile_int + 1 

        repair_action = curr_tile_int

        return repair_action

    
class PoDRepresentationWrapper(RepresentationWrapper):
    # FIXME: Move to (narrow) representation.
    def __init__(self, rep, cfg):
        super().__init__(rep, cfg)
        self.rep.unwrapped.get_act_coords = self.get_act_coords

    def get_act_coords(self):
        act_coords = NarrowRepresentation.get_act_coords(self.rep)
        act_coords = np.flip(act_coords, axis=0)
        return act_coords