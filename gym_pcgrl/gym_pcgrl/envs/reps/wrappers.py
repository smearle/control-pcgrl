from collections import OrderedDict
from pdb import set_trace as TT

from gym import spaces
import numpy as np
from PIL import Image

from gym_pcgrl.envs.reps.ca_holey import CARepresentationHoley
from gym_pcgrl.envs.reps.ca_rep import CARepresentation
from gym_pcgrl.envs.reps.narrow_holey_rep import NarrowHoleyRepresentation
from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from gym_pcgrl.envs.reps.representation import Representation


def wrap_holey(rep_cls):
    # TODO: Not finished implementing this. Not using for now.

    class HoleyRepresentation(rep_cls):

        def reset(self, *args, **kwargs):
            ret = rep_cls.reset(self, *args, **kwargs)
            rep_cls.reset(self)
            return ret

        def get_observation(self):
            obs = rep_cls.get_observation(self)
            obs.update(
                {'map': self._bordered_map(),}
            )
            return obs

        def get_observation_space(self, width, height, num_tiles):
            obs_space = rep_cls.get_observation_space(self, width, height, num_tiles)
            obs_space.spaces.update({
                "map": spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=(height+2, width+2))
            })
            return obs_space

        if issubclass(rep_cls, NarrowRepresentation):
            render = NarrowHoleyRepresentation.render

    TT()

    return HoleyRepresentation


def wrap_static_build(rep_cls):

    class StaticBuildRepresentation(rep_cls):
        def __init__(self, *args, **kwargs):
            rep_cls.__init__(self, *args, **kwargs)
            self.prob_static = 0.0
            self.window = None

        def adjust_param(self, **kwargs):
            self.prob_static = kwargs.get('static_prob')
            self.n_aux_tiles = kwargs.get('n_aux_tiles')
            return rep_cls.adjust_param(self, **kwargs)

        def reset(self, *args, **kwargs):
            ret = rep_cls.reset(self, *args, **kwargs)
            # TODO: take into account validity constraints on number of certain tiles
            self.static_builds = (np.random.random(self._bordered_map.shape) < self.prob_static).astype(np.uint8)
            # Borders are always static
            self.static_builds[(0, -1), :] = 1
            self.static_builds[:, (0, -1)] = 1

            # Remove any action coordinates that correspond to static tiles (unless we have aux chans, in which case 
            # we'll let the agent leave messages for itself on those channels, even on static tiles.)
            # NOTE: We only have `_act_coords` for narrow representation. Can we make this cleaner?
            if hasattr(self, '_act_coords') and self.n_aux_tiles == 0:
                self._act_coords = self._act_coords[np.where(
                    self.static_builds[self._act_coords[:, 0], self._act_coords[:, 1]] == 0)] 
            return ret

        def get_observation_space(self, width, height, num_tiles):
            obs_space = rep_cls.get_observation_space(self, width, height, num_tiles)
            obs_space.spaces.update({
                'static_builds': spaces.Box(low=0, high=1, dtype=np.uint8, shape=(height, width))
            })
            return obs_space

        def get_observation(self):
            obs = rep_cls.get_observation(self)
            obs.update({
                'static_builds': self.static_builds,
            })
            return obs

        def render(self, lvl_image, tile_size, border_size=None):
            lvl_image = rep_cls.render(self, lvl_image, tile_size, border_size)
            x_graphics = Image.new("RGBA", (tile_size,tile_size), (0,0,0,0))
            for x in range(tile_size):
                x_graphics.putpixel((0,x),(255,0,0,255))
                x_graphics.putpixel((1,x),(255,0,0,255))
                x_graphics.putpixel((tile_size-2,x),(255,0,0,255))
                x_graphics.putpixel((tile_size-1,x),(255,0,0,255))
            for y in range(tile_size):
                x_graphics.putpixel((y,0),(255,0,0,255))
                x_graphics.putpixel((y,1),(255,0,0,255))
                x_graphics.putpixel((y,tile_size-2),(255,0,0,255))
                x_graphics.putpixel((y,tile_size-1),(255,0,0,255))
            for (y, x) in np.argwhere(self.static_builds[1:-1, 1:-1] == 1):
                y, x = y + 1, x + 1  # ignoring the border
                lvl_image.paste(x_graphics, ((x+border_size[0]-1)*tile_size, (y+border_size[1]-1)*tile_size,
                                                (x+border_size[0])*tile_size,(y+border_size[1])*tile_size), x_graphics)

            # if not hasattr(self, 'window'):
                # self.window = cv2.namedWindow('static builds', cv2.WINDOW_NORMAL)
                # cv2.resize('static builds', 100, 800)
                # cv2.waitKey(1)
            # im = self.static_builds.copy()
            # cv2.imshow('static builds', im * 255)
            # cv2.waitKey(1)

            return lvl_image

        # update = {
            # CARepresentationHoley: update_ca_holey,
        # }[rep_cls]

        def update(self, action, **kwargs):
            old_state = self._bordered_map.copy()
            change, pos = rep_cls.update(self, action, **kwargs)
            new_state = self._bordered_map
            # assert not(np.all(old_state == new_state))
            self._bordered_map = np.where(self.static_builds < 1, new_state, old_state)
            # print(self._bordered_map)
            self._map = self._bordered_map[1:-1, 1:-1]
            change = np.any(old_state != new_state)
            return change, pos

    return StaticBuildRepresentation


# def update_ca_holey(self, action, **kwargs):
#     old_state = self._bordered_map.copy()
#     change, pos = CARepresentationHoley.update(self, action, **kwargs)
#     new_state = self._bordered_map
#     # assert not(np.all(old_state == new_state))
#     self._bordered_map = np.where(self.static_builds < 1, new_state, old_state)
#     # print(self._bordered_map)
#     self._map = self._bordered_map[1:-1, 1:-1]
#     change = np.any(old_state != new_state)
#     return change, pos


# def update_narrow_holey(self, action, **kwargs):
#     change = 0
#     if action > 0:
#         change += [0,1][self._map[self._y][self._x] != action-1]
#         self._map[self._y][self._x] = action-1
#         self._bordered_map[self._y+1][self._x+1] = action-1
#     if self._random_tile:
#         if self.n_step == len(self._act_coords):
#             np.random.shuffle(self._act_coords)
#     self._x, self._y = self._act_coords[self.n_step % len(self._act_coords)]
#     self.n_step += 1
#     return change, [self._x, self._y]