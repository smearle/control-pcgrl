################################################################################
#   Conditional Wrapper
################################################################################
import collections
import copy
import json
from pdb import set_trace as TT
from timeit import default_timer as timer
from typing import Dict, OrderedDict

# import gi 
# gi.require_version("Gtk", "3.0")
# from gi.repository import Gtk
import gymnasium as gym
import numpy as np

from control_pcgrl.configs.config import Config
# from opensimplex import OpenSimplex
from control_pcgrl.envs.helper import get_range_reward
import ray


# TODO: Make this part of the PcgrlEnv class instead of a wrapper?
# FIXME: This is not calculating the loss from a metric value (point) to a target metric range (line) correctly.
# In particular we're only looking at integers and we're excluding the upper bound of the range.
class ControlWrapper(gym.Wrapper):
    def __init__(self, env, cfg: Config, ctrl_metrics=None, rand_params=False):

        # Is this a controllable agent? If false, we're just using this wrapper for convenience, to calculate relative
        # reward and establish baseline performance
        self.controllable = ctrl_metrics is not None

        # We'll use these for calculating loss (for evaluation during inference) but not worry about observing them
        # (we didn't necessarily train with all of them)
        ctrl_loss_metrics = ctrl_metrics if ctrl_metrics is not None else []

        self.render_gui = cfg.render
        self.env = env
        super().__init__(self.env)

        metric_weights = copy.copy(self.unwrapped._reward_weights)
        self.metric_weights = {k: 0 for k in metric_weights}
        config_weights = cfg.task.weights

        self.metric_weights.update(config_weights)

        # fixed metrics (i.e. playability constraints)
        self.static_metric_names = set(env.static_trgs.keys())

        for k in ctrl_loss_metrics:
            if k in self.static_metric_names:
                self.static_metric_names.remove(k)
        self.auto_reset = True

        # NOTE: assign self.metrics after having the underlying env get its _rep_stats, or they will be behind.
        self.metrics = self.unwrapped.metrics

        # NB: self.metrics needs to be an OrderedDict
        # print("usable metrics for conditional wrapper:", self.ctrl_metrics)
        # print("unwrapped env's current metrics: {}".format(self.unwrapped.metrics))
        self.last_metrics = copy.deepcopy(self.metrics)
        self.cond_bounds = self.unwrapped.cond_bounds
        self.param_ranges = {}

        # controllable metrics
        self.ctrl_metrics = ctrl_metrics if ctrl_metrics is not None else [] 
        self.n_ctrl_metrics = len(self.ctrl_metrics)

        for k in self.ctrl_metrics:
            v = self.cond_bounds[k]
            improvement = abs(v[1] - v[0])
            self.param_ranges[k] = improvement

        # We might be using a subset of possible conditional targets supplied by the problem

        self.metric_trgs = self.unwrapped.static_trgs

        # All metrics that we will consider for reward
        self.all_metrics = set()
        self.all_metrics.update(self.ctrl_metrics)
        self.all_metrics.update(ctrl_loss_metrics)  # probably some overlap here
        self.all_metrics.update(self.static_metric_names)

        self.n_metrics = len(self.all_metrics)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # TODO: generalize this for 3D environments.
        if self.controllable:
            orig_obs_shape = self.observation_space.shape
            # TODO: adapt to (c, w, h) vs (w, h, c)

            # We will add channels to the observation space for each controllable metric.
            self.n_new_obs = n_new_obs = len(self.ctrl_metrics) * 2

            self.CHAN_LAST = False
            obs_shape = (
                *orig_obs_shape[:-1],
                orig_obs_shape[-1] + n_new_obs,
            )
            low = self.observation_space.low
            high = self.observation_space.high
            metrics_shape = (*obs_shape[:-1], n_new_obs)
            self.metrics_shape = metrics_shape
            metrics_low = np.full(metrics_shape, fill_value=0)
            metrics_high = np.full(metrics_shape, fill_value=1)
            low = np.concatenate((metrics_low, low), axis=-1)
            high = np.concatenate((metrics_high, high), axis=-1)
            self.observation_space = gym.spaces.Box(low=low, high=high)

        # Does this need to be a queue? Do we ever queue up more than one set of targets?
        self._ctrl_trg_queue = []

        self.infer = cfg.infer
        self.last_loss = None
        self.ctrl_loss_metrics = ctrl_loss_metrics
        self.max_loss = self.get_max_loss(ctrl_metrics=ctrl_loss_metrics)

        if self.render_mode == 'gtk':
            self.win = self._init_gui()


    def _init_gui(self):
        screen_width = 200
        screen_height = 100 * self.n_metrics
        from control_pcgrl.gtk_gui import GtkGUI

        if not hasattr(self.unwrapped._prob, "_graphics"):
            _graphics = {}
        elif self.unwrapped._prob._graphics is None:
            self.unwrapped._prob.init_graphics()
            _graphics = self.unwrapped._prob._graphics
        win = GtkGUI(env=self, tile_types=self.unwrapped._prob.get_tile_types(), tile_images=_graphics, 
            metrics=self.metrics, metric_trgs=self.metric_trgs, metric_bounds=self.cond_bounds)
        # win.connect("destroy", Gtk.main_quit)
        win.show_all()
        win = win
        return win

    def get_control_bounds(self):
        controllable_bounds = {k: self.cond_bounds[k] for k in self.ctrl_metrics}

        return controllable_bounds

    def get_control_vals(self):
        control_vals = {k: self.metrics[k] for k in self.ctrl_metrics}

        return control_vals

    def get_metric_vals(self):
        return self.metrics

    def toggle_auto_reset(self, button):
        self.auto_reset = button.get_active()

    def getState(self):
        scalars = super().getState()
        print("scalars: ", scalars)
        raise Exception

        return scalars

    def queue_control_trgs(self, idx_counter):
        self._ctrl_trg_queue = ray.get(idx_counter.get.remote(hash(self)))

    def set_trgs(self, trgs):
        self._ctrl_trg_queue = [trgs]

    def do_set_trgs(self, trgs):
        self.metric_trgs.update(trgs)
        self.display_metric_trgs()

    def reset(self, *, seed=None, options=None):
        if len(self._ctrl_trg_queue) > 0:
            next_trgs = self._ctrl_trg_queue.pop(0)
            self.do_set_trgs(next_trgs)
        ob, info = super().reset()
        self.metrics = self.unwrapped._rep_stats
        if self.controllable:
            ob = self.observe_metric_trgs(ob)
        self.last_metrics = copy.deepcopy(self.metrics)
        # if self.unwrapped._get_stats_on_step:
        self.last_loss = self.get_loss()
        self.n_step = 0

        return ob, info

    def observe_metric_trgs(self, obs):
        # metrics_ob = np.zeros(self.n_new_obs)
        metrics_ob = np.zeros(self.metrics_shape)

        i = 0

        for k in self.ctrl_metrics:
            trg = self.metric_trgs[k]
            metric = self.metrics[k]

            if not metric:
                metric = 0
            trg_range = self.param_ranges[k]

            if isinstance(trg, tuple):
                trg = (trg[0] + trg[1]) / 2

            # Add channel layers filled with scalar values corresponding to the target values of metrics of interest.
            metrics_ob[:, :, i*2] = trg / self.param_ranges[k]
            metrics_ob[:, :, i*2+1] = metric / self.param_ranges[k]
            i += 1

        obs = np.concatenate((metrics_ob, obs), axis=-1)
        # TODO: support dictionary observations?

        return obs

    def step(self, action, **kwargs):
        ob, rew, done, truncated, info = super().step(action, **kwargs)
        self.metrics = self.unwrapped._rep_stats

        # Add target values of metrics of interest to the agent's obervation, so that it can learn to reproduce them 
        # while editing the level. 
        if self.controllable:
            ob = self.observe_metric_trgs(ob)

        # Provide reward only at the last step

        loss = self.get_loss()
        reward = loss - self.last_loss
        self.last_loss = loss

        self.last_metrics = self.metrics
        self.last_metrics = copy.deepcopy(self.metrics)
        self.n_step += 1

        # This should only happen during inference, when user is interacting with GUI.
        if not self.auto_reset:
            done = False
            truncated = False

        if self.render_mode in {'human', 'gtk', 'save_gif'}:
            self.render()

        # print("step: ", self.n_step, " done: ", done, " reward: ", reward, " action: ", action, " metrics: ", self.metrics)
        return ob, reward, done, truncated, info

    def render(self, **kwargs):
        if self.render_mode == 'gtk':
            img = super().render()
            self.win.render(img)
            user_clicks = self.win.get_clicks()
            for (py, px, tile, static) in user_clicks:

                # FIXME: this logic belongs inside gtk_gui...?
                # First subtract the pygtk EventBox border size
                # Size of rendered map in pixels
                map_size_pix = self.unwrapped._prob._tile_size * (np.array(self.unwrapped._rep.unwrapped._map.shape) + np.array(self.unwrapped._prob._border_size) * 2)

                # Assume map is always rendered in the middle of its eventbox
                widget_border_size = (np.array([self.win.map_eventbox.get_allocated_width(), self.win.map_eventbox.get_allocated_height()]) - map_size_pix) / 2
                px, py = px - widget_border_size[1], py - widget_border_size[0]

                x = int(px // self.unwrapped._prob._tile_size - self.unwrapped._prob._border_size[1])
                y = int(py // self.unwrapped._prob._tile_size - self.unwrapped._prob._border_size[0])
                if x < 0 or y < 0 or x >= self.unwrapped._rep.unwrapped._map.shape[0] or y >= self.unwrapped._rep.unwrapped._map.shape[1]:
                    print("Clicked outside of map")
                    continue

                tile_int = self.unwrapped._prob.get_tile_int(tile)
                print(f"Place tile {tile} at x:{x}, y:{y}")

                # FIXME: This is a hack. Write a function in Representation for this.
                self.unwrapped._rep.unwrapped._map[x, y] = tile_int

                if hasattr(self.unwrapped._rep, 'static_builds'):
                    # Assuming borders of width 1 (different than `_border_size` above, which may be different for rendering purposes).
                    self.unwrapped._rep.static_builds[x+1, y+1] = int(static)

            self.unwrapped._rep.unwrapped._update_bordered_map()

            if self.win._paused:
                self.render()

        elif self.render_mode == 'save_gif':
            img = super().render()
            return img

        else:
            ### PROFILING ###
            if kwargs.get("render_profiling"):
                N = 100
                start_time = timer()
                for _ in range(N):
                    super().reset()
                    super().render()
                print(f'mean pyglet image render time over {N} trials:', (timer() - start_time) * 1000 / N, 'ms')

                start_time = timer()
                for _ in range(N):
                    img = super().render()
                    self.win.render(img)
                print(f'mean pygobject image render time over {N} trials:', (timer() - start_time) * 1000 / N, 'ms')
            ### END PROFILING ###
 
            return super().render()

    def get_cond_trgs(self):
        return self.metric_trgs

    def get_cond_bounds(self):
        return self.cond_bounds

    def display_metric_trgs(self):
        if self.render_gui:
            # if self.win is None:
                # self._init_gui()
            self.win.display_metric_trgs()

    def get_loss(self):
        """Get the distance between the current values of the metrics and their targets. Note that this means we need
        to fix finite bounds on the values of all metrics."""
        loss = 0

        for metric in self.all_metrics:
            if metric in self.metric_trgs:
                trg = self.metric_trgs[metric]
            else:
                trg = self.static_trgs[metric]

            val = self.metrics[metric]

            if isinstance(trg, tuple):
                # then we assume it corresponds to a target range, and we penalize the minimum distance to that range
                loss_m = -abs(np.arange(*trg) - val).min()
            else:
                loss_m = -abs(trg - val)
            loss_m = loss_m * self.metric_weights[metric]
            loss += loss_m

        return loss

    def get_max_loss(self, ctrl_metrics=[]):
        '''Upper bound on distance of level from static targets.
        
        Args:
            ctrl_metrics (list): list of metrics to be controlled (in RL), or used as diversity measured (in QD). These
                will not factor into the loss.
        '''
        net_max_loss = 0
        for k, v in self.static_trgs.items():
            if k in ctrl_metrics:
                continue
            if isinstance(v, tuple):
                max_loss = max(abs(v[0] - self.cond_bounds[k][0]), abs(v[1] - self.cond_bounds[k][1])) * self.metric_weights[k]
            else: max_loss = max(abs(v - self.cond_bounds[k][0]), abs(v - self.cond_bounds[k][1])) * self.metric_weights[k]
            net_max_loss += max_loss
        return net_max_loss

    def get_ctrl_loss(self):
        """How far away (L1) from we from our vector of control targets?"""
        loss = 0

        for metric in self.ctrl_loss_metrics:
            trg = self.metric_trgs[metric]
            val = self.metrics[metric]
            if isinstance(trg, tuple):
                loss_m = -abs(np.arange(*trg) - val).min()
            else:
                loss_m = -abs(trg - val)
            loss_m = loss_m * self.metric_weights[metric]
            loss += loss_m

        return loss

    # def get_done(self):
    #     done = True
    #     # overwrite static trgs with conditional ones, in case we have made a static one conditional in this run
    #     trg_dict = copy.deepcopy(self.static_trgs)
    #     trg_dict.update(self.metric_trgs)

    #     for k, v in trg_dict.items():
    #         if isinstance(v, tuple):
    #             # FIXME: Missing the upper bound here!
    #             if self.metrics[k] in np.arange(*v):
    #                 done = False
    #         elif int(self.metrics[k]) != int(v):
    #             done = False

    #     if done and self.infer:
    #         print("targets reached! {}".format(trg_dict))

    #     return done

    def close(self):
        if self.render_gui and self.controllable:
            self.win.destroy()


# TODO: What by jove this actually doing and why does it kind of work?
# class PerlinNoiseyTargets(gym.Wrapper):
#    '''A bunch of simplex noise instances modulate target metrics.'''
#    def __init__(self, env, **kwargs):
#        super(PerlinNoiseyTargets, self).__init__(env)
#        self.cond_bounds = self.env.unwrapped.cond_bounds
#        self.num_params = self.num_params
#        self.noise = OpenSimplex()
#        # Do not reset n_step so that we keep moving through the perlin noise and do not repeat our course
#        self.n_step = 0
#        self.X, self.Y = np.random.random(2) * 10000
#
#    def step(self, a):
#        cond_bounds = self.cond_bounds
#        trgs = {}
#        i = 0
#
#        for k in self.env.ctrl_metrics:
#            trgs[k] = self.noise.noise2d(x=self.X + self.n_step/400, y=self.Y + i*100)
#            i += 1
#
#        i = 0
#
#        for k in self.env.ctrl_metrics:
#            (ub, lb) = cond_bounds[k]
#            trgs[k] = ((trgs[k] + 1) / 2 * (ub - lb)) + lb
#            i += 1
#        self.env.set_trgs(trgs)
#        out = self.env.step(a)
#        self.n_step += 1
#
#        return out
#
#    def reset(self, *, seed=None, options=None):
##       self.noise = OpenSimplex()
#        return self.env.reset()


class UniformNoiseyTargets(gym.Wrapper):
    """A bunch of simplex noise instances modulate target metrics."""

    def __init__(self, env, cfg: Config):
        super(UniformNoiseyTargets, self).__init__(env)
        self.cond_bounds = self.env.unwrapped.cond_bounds
        self.num_params = self.num_params
        # self.midep_trgs = kwargs.get("midep_trgs", False)
        self.midep_trgs = False

    def set_rand_trgs(self):
        trgs = {}

        for k in self.env.ctrl_metrics:
            (lb, ub) = self.cond_bounds[k]
            trgs[k] = np.random.random() * (ub - lb) + lb
        self.env.set_trgs(trgs)

    def step(self, action, **kwargs):
        if self.midep_trgs:
            if np.random.random() < 0.005:
                self.set_rand_trgs()
                self.do_set_trgs()

        return self.env.step(action, **kwargs)

    def reset(self, *, seed=None, options=None):
        self.set_rand_trgs()

        return self.env.reset(), {}


# class ALPGMMTeacher(gym.Wrapper):
#     def __init__(self, env, **kwargs):

#         from teachDRL.teachers.algos.alp_gmm import ALPGMM

#         super(ALPGMMTeacher, self).__init__(env)
#         self.cond_bounds = self.env.unwrapped.cond_bounds
#         self.midep_trgs = False
#         env_param_lw_bounds = [self.cond_bounds[k][0] for k in self.ctrl_metrics]
#         env_param_hi_bounds = [self.cond_bounds[k][1] for k in self.ctrl_metrics]
#         self.alp_gmm = ALPGMM(env_param_lw_bounds, env_param_hi_bounds)
#         self.trg_vec = None
#         self.trial_reward = 0
#         self.n_trial_steps = 0

#     def reset(self, *, seed=None, options=None):
#         if self.trg_vec is not None:
#             if self.n_trial_steps == 0:
#                 # This is some whackness that happens when we reset manually from the inference script.
#                 rew = 0
#             else:
#                 rew = self.trial_reward / self.n_trial_steps
#             self.alp_gmm.update(self.trg_vec, rew)
#         trg_vec = self.alp_gmm.sample_task()
#         self.trg_vec = trg_vec
#         trgs = {k: trg_vec[i] for (i, k) in enumerate(self.ctrl_metrics)}
#         #       print(trgs)
#         self.set_trgs(trgs)
#         self.trial_reward = 0
#         self.n_trial_steps = 0

#         return self.env.reset()

#     def step(self, action, **kwargs):
#         obs, rew, done, truncated, info = self.env.step(action, **kwargs)
#         self.trial_reward += rew
#         self.n_trial_steps += 1

#         return obs, rew, done, truncated, info
