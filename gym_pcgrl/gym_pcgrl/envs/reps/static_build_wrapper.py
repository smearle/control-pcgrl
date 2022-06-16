from pdb import set_trace as TT

from gym import spaces
import numpy as np
from PIL import Image

from gym_pcgrl.envs.reps.ca_holey import CARepresentationHoley
from gym_pcgrl.envs.reps.ca_rep import CARepresentation
from gym_pcgrl.envs.reps.representation import Representation


def wrap_static_build(rep_cls):

    class StaticBuildRepresentation(rep_cls):
        def __init__(self, *args, **kwargs):
            rep_cls.__init__(self, *args, **kwargs)
            self.prob_static = 0.0
            self.window = None

        def adjust_param(self, **kwargs):
            self.prob_static = kwargs.get('prob_static', 0.1)
            return rep_cls.adjust_param(self, **kwargs)

        def reset(self, *args, **kwargs):
            ret = rep_cls.reset(self, *args, **kwargs)
            # TODO: take into account validity constraints on number of certain tiles
            self.static_builds = (np.random.random(self._bordered_map.shape) < self.prob_static).astype(np.uint8)
            # Borders are always static
            self.static_builds[(0, -1), :] = 1
            self.static_builds[:, (0, -1)] = 1
            return ret

        def get_observation_space(self, width, height, num_tiles):
            obs_space = rep_cls.get_observation_space(self, width, height, num_tiles)
            obs_space.spaces.update({
                'static_builds': spaces.Box(low=0, high=1, dtype=np.uint8, shape=(height, width))
            })
            return obs_space

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
            for (y, x) in np.argwhere(self.static_builds == 1):
                lvl_image.paste(x_graphics, ((x)*tile_size, (y)*tile_size,
                                                (x+1)*tile_size,(y+1)*tile_size), x_graphics)

            # if not hasattr(self, 'window'):
                # self.window = cv2.namedWindow('static builds', cv2.WINDOW_NORMAL)
                # cv2.resize('static builds', 100, 800)
                # cv2.waitKey(1)
            # im = self.static_builds.copy()
            # cv2.imshow('static builds', im * 255)
            # cv2.waitKey(1)

            return lvl_image

        update = {
            CARepresentationHoley: update_ca_holey,
        }[rep_cls]

    return StaticBuildRepresentation


def update_ca_holey(self, action, **kwargs):
    old_state = self._bordered_map.copy()
    change, pos = CARepresentationHoley.update(self, action, **kwargs)
    new_state = self._bordered_map
    # assert not(np.all(old_state == new_state))
    self._bordered_map = np.where(self.static_builds < 1, new_state, old_state)
    # print(self._bordered_map)
    self._map = self._bordered_map[1:-1, 1:-1]
    change = np.any(old_state != new_state)
    return change, pos

