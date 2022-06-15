from pdb import set_trace as TT
from typing import List
import cv2

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from PIL import Image

from gym_pcgrl.envs.helper import gen_random_map
from gym_pcgrl.envs.reps.representation import Representation


class HoleyRepresentation(Representation):
    def reset(self, *args, **kwargs):
        self.dig_holes(self.entrance_coords, self.exit_coords)

    def set_holes(self, entrance_coords, exit_coords):
        self.entrance_coords, self.exit_coords = entrance_coords, exit_coords

    def dig_holes(self, entrance_coords, exit_coords):
        # TODO: Represent start/end differently to accommodate one-way paths.
        self._bordered_map[entrance_coords[0], entrance_coords[1]] = self._empty_tile
        self._bordered_map[exit_coords[0], exit_coords[1]] = self._empty_tile


class StaticBuildRepresentation():
    def __init__(self):
        self.prob_static = 0.0
        self.window = None

    def adjust_param(self, **kwargs):
        self.prob_static = kwargs.get('prob_static', 0.1)
        return super().adjust_param(**kwargs)

    def reset(self, *args, **kwargs):
        # TODO: take into account validity constraints on number of certain tiles
        self.static_builds = (np.random.random(self._bordered_map.shape) < self.prob_static).astype(np.uint8)
        # Borders are always static
        self.static_builds[(0, -1), :] = 1
        self.static_builds[:, (0, -1)] = 1
        

    def get_observation_space(self, width, height, num_tiles):
        return spaces.Dict({
            'static_builds': spaces.Box(low=0, high=1, dtype=np.uint8, shape=(height, width))
        })


    def render(self, lvl_image, tile_size, border_size=None):
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
    # def render(self, *args, **kwargs):
        if not hasattr(self, 'window'):
            self.window = cv2.namedWindow('static builds', cv2.WINDOW_NORMAL)
            # cv2.resize('static builds', 100, 800)
            cv2.waitKey(1)
        im = self.static_builds.copy()
        cv2.imshow('static builds', im * 255)
        cv2.waitKey(1)

        return lvl_image