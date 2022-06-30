################################################################################
#   Conditional Wrapper
################################################################################
import collections
import copy
from pdb import set_trace as TT
from typing import Dict, OrderedDict

import gym
import numpy as np

# from opensimplex import OpenSimplex
from gym_pcgrl.envs.helper import get_range_reward
import ray


#FIXME: This is not calculating the loss from a metric value (point) to a target metric range (line) correctly.
# In particular we're only looking at integers and we're excluding the upper bound of the range.
class ConditionalWrapper(gym.Wrapper):
    def __init__(self, env, ctrl_metrics=[], rand_params=False, **kwargs):
        # Is this a controllable agent? If false, we're just using this wrapper for convenience, to calculate relative
        # reward and establish baseline performance
        self.conditional = kwargs.get("conditional")
        # We'll use these for calculating loss (for evaluation during inference) but not worry about oberving them
        # (we didn't necessarily train with all of them)
        ctrl_loss_metrics = kwargs.get("eval_controls")
        if not ctrl_loss_metrics:
            ctrl_loss_metrics = ctrl_metrics if ctrl_metrics is not None else []
        if self.conditional:
            assert ctrl_metrics
        else:
            if ctrl_metrics:
                print('Dummy controllable metrics: {}, will not be observed.'.format(ctrl_metrics))
        self.CA_action = kwargs.get("ca_action")

        self.render_gui = kwargs.get("render") and self.conditional
        # Whether to always select random parameters, to stabilize learning multiple objectives
        # (i.e. prevent overfitting to some subset of objectives)
        #       self.rand_params = rand_params
        self.env = env
        super().__init__(self.env)
        self.metric_weights = copy.copy(self.unwrapped._reward_weights)
        #       cond_trgs = self.unwrapped.cond_trgs

        self.ctrl_metrics = ctrl_metrics  # controllable metrics
#       self.ctrl_metrics = ctrl_metrics[::-1] # Fucking stupid bug resulting from a fix introduced partway through training a relevant batch of experiments. Delete this in favor of line above eventually.
        # fixed metrics (i.e. playability constraints)
        self.static_metric_names = set(env.static_trgs.keys())

        for k in ctrl_loss_metrics:
            if k in self.static_metric_names:
                self.static_metric_names.remove(k)
        self.num_params = len(self.ctrl_metrics)
        self.auto_reset = True
        # self.unwrapped._reward_weights = {}

        #       self.unwrapped.configure(**kwargs)

        # NOTE: assign self.metrics after having the underlying env get its _rep_stats, or they will be behind.
        self.metrics = self.unwrapped.metrics

        # NB: self.metrics needs to be an OrderedDict
        # print("usable metrics for conditional wrapper:", self.ctrl_metrics)
        # print("unwrapped env's current metrics: {}".format(self.unwrapped.metrics))
        self.last_metrics = copy.deepcopy(self.metrics)
        self.cond_bounds = self.unwrapped.cond_bounds
        self.param_ranges = {}
        #       self.max_improvement = 0

        for k in self.ctrl_metrics:
            v = self.cond_bounds[k]
            improvement = abs(v[1] - v[0])
            self.param_ranges[k] = improvement
            self.metric_weights[k] = self.unwrapped._ctrl_reward_weights[k]
        #           self.max_improvement += improvement * self.unwrapped._reward_weights[k]

        self.metric_trgs = {}
        # we might be using a subset of possible conditional targets supplied by the problem

        for k in self.ctrl_metrics:
            self.metric_trgs[k] = self.unwrapped.static_trgs[k]

        # All metrics that we will consider for reward
        self.all_metrics = set()
        self.all_metrics.update(self.ctrl_metrics)
        self.all_metrics.update(ctrl_loss_metrics)  # probably some overlap here
        self.all_metrics.update(self.static_metric_names)

        if "RCT" in str(type(env.unwrapped)) or "Micropolis" in str(
            type(env.unwrapped)
        ):
            self.SC_RCT = True
        else:
            self.SC_RCT = False

        for k in self.all_metrics:
            v = self.metrics[k]

            # if self.SC_RCT and k not in self.ctrl_metrics:
                # self.unwrapped._reward_weights[k] = 0
            # else:
                # self.unwrapped._reward_weights[k] = self.unwrapped._reward_weights[k]

        #       for k in self.ctrl_metrics:
        #           self.cond_bounds['{}_weight'.format(k)] = (0, 1)
        self.width = self.unwrapped.width
        self.observation_space = self.env.observation_space
        # FIXME: hack for gym-pcgrl
        # print("conditional wrapper, original observation space shape", self.observation_space.shape)
        self.action_space = self.env.action_space

        # TODO: generalize this for 3D environments.
        if self.conditional:
            orig_obs_shape = self.observation_space.shape
            # TODO: adapt to (c, w, h) vs (w, h, c)

            # We will add channels to the observation space for each controllable metric.
            # if self.CA_action:
                #       if self.CA_action and False:
                # n_new_obs = 2 * len(self.ctrl_metrics)
            #           n_new_obs = 1 * len(self.ctrl_metrics)
            # else:
                # n_new_obs = 1 * len(self.ctrl_metrics)
            self.n_new_obs = n_new_obs = len(self.ctrl_metrics) * 2

            if self.SC_RCT:
                pass
                # self.CHAN_LAST = True
                # obs_shape = (
                #     *orig_obs_shape[1:],
                #     orig_obs_shape[0] + n_new_obs,
                # )
                # low = self.observation_space.low.transpose(1, 2, 0)
                # high = self.observation_space.high.transpose(1, 2, 0)
            else:
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

            # if not isinstance(self.observation_space, gym.spaces.Dict):
            #     obs_space = gym.spaces.Dict(
            #     map=self.observation_space,
            #     ctrl_metrics=gym.spaces.Box(
            #             low=0, high=1, shape=(self.n_new_obs,), dtype=np.float32))
            #     obs_space.dtype = collections.OrderedDict
            #     self.observation_space = obs_space
            # else:
            #     raise Exception

        self.next_trgs = None
        self._ctrl_trg_queue = []

        if self.render_gui and self.conditional:
            screen_width = 200
            screen_height = 100 * self.num_params
            from gym_pcgrl.conditional_window import ParamRewWindow

            win = ParamRewWindow(self, self.metrics, self.metric_trgs, self.cond_bounds)
            # win.connect("destroy", Gtk.main_quit)
            win.show_all()
            self.win = win
        self.infer = kwargs.get("infer", False)
        self.last_loss = None
        self.ctrl_loss_metrics = ctrl_loss_metrics
        self.max_loss = self.get_max_loss(ctrl_metrics=ctrl_loss_metrics)


    def get_control_bounds(self):
        controllable_bounds = {k: self.cond_bounds[k] for k in self.ctrl_metrics}

        return controllable_bounds

    def get_control_vals(self):
        control_vals = {k: self.metrics[k] for k in self.ctrl_metrics}

        return control_vals

    def get_metric_vals(self):
        return self.metrics

    def configure(self, **kwargs):
        pass

    def enable_auto_reset(self, button):
        self.auto_reset = button.get_active()

    def getState(self):
        scalars = super().getState()
        print("scalars: ", scalars)
        raise Exception

        return scalars

    def queue_control_trgs(self, idx_counter):
        self._ctrl_trg_queue = ray.get(idx_counter.get.remote(hash(self)))

    def set_trgs(self, trgs):
        self.next_trgs = trgs

    def do_set_trgs(self):
        trgs = self.next_trgs
        i = 0
        self.init_metrics = copy.deepcopy(self.metrics)

        for k, trg in trgs.items():
            # Here we will allow setting targets that are not controlled for in training.
            # These will not be observed, but will factor into reward.
            #           if k in self.ctrl_metrics:
            self.metric_trgs[k] = trg

        self.display_metric_trgs()

    def reset(self):
        if len(self._ctrl_trg_queue) > 0:
            self.next_trgs, self._ctrl_trg_queue = self._ctrl_trg_queue[0], self._ctrl_trg_queue[1:]
        if self.next_trgs:
            self.do_set_trgs()
        ob = super().reset()
        self.metrics = self.unwrapped._rep_stats
        if self.conditional:
            ob = self.observe_metric_trgs(ob)
        self.last_metrics = copy.deepcopy(self.metrics)
        if self.unwrapped._get_stats_on_step:
            self.last_loss = self.get_loss()
        self.n_step = 0

        return ob

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
            #           if self.CA_action and False:

            if isinstance(trg, tuple):
                trg = (trg[0] + trg[1]) / 2

            # CA actions for RL agent are not implemented
            # if self.CA_action:
                #               metrics_ob[:, :, i] = (trg - metric) / trg_range
                # pass
            # metrics_ob[..., i * 2] = trg / self.param_ranges[k]
            # metrics_ob[..., i * 2 + 1] = metric / self.param_ranges[k]

            # else:
            # metrics_ob[i] = trg / trg_range
            # metrics_ob[i * 2 + 1] = metric / trg_range

                # Formerly was showing the direction of desired change with current values updated at each step.
                # metrics_ob[:, :, i] = np.sign(trg / trg_range - metric / trg_range)

            # Add channel layers filled with scalar values corresponding to the target values of metrics of interest.
            metrics_ob[:, :, i*2] = trg / self.param_ranges[k]
            metrics_ob[:, :, i*2+1] = metric / self.param_ranges[k]
            i += 1
        #       print('param rew obs shape ', obs.shape)
        #       print('metric trgs shape ', metrics_ob.shape)
        #       if self.CHAN_LAST:
        #           obs = obs.transpose(1, 2, 0)
        obs = np.concatenate((metrics_ob, obs), axis=-1)

        # TODO: support dictionary observations?
        # assert isinstance(obs, np.ndarray)
        # obs = {'map': obs, 'ctrl_metrics': metrics_ob}
        return obs

    def step(self, action, **kwargs):
        if self.render_gui:
            self.win.step()

        ob, rew, done, info = super().step(action, **kwargs)
        self.metrics = self.unwrapped._rep_stats

        # Add target values of metrics of interest to the agent's obervation, so that it can learn to reproduce them 
        # while editing the level. 
        if self.conditional:
            ob = self.observe_metric_trgs(ob)

        # Provide reward only at the last step
        reward = self.get_reward()  # if done else 0

        self.last_metrics = self.metrics
        self.last_metrics = copy.deepcopy(self.metrics)
        self.n_step += 1

        if self.auto_reset:
            # either exceeded number of changes, steps, or reached target

            # I have no idea wtf `get_done()` is for. Maybe early termination during evaluation, where we give the agent
            # a ton of steps to reach its goals? Omitting for now.
            # done = done or self.get_done()
            done = done
        else:
#           assert self.infer
            done = False


        return ob, reward, done, info

    def get_cond_trgs(self):
        return self.metric_trgs

    def get_cond_bounds(self):
        return self.cond_bounds

    def set_cond_bounds(self, bounds):
        for k, (l, h) in bounds.items():
            self.cond_bounds[k] = (l, h)

    def display_metric_trgs(self):
        if self.render_gui:
            self.win.display_metric_trgs()

    def get_loss(self):
        """Get the distance between the current values of the metrics and their targets. Note that this means we need
        to fix finite bounds on the values of all metrics."""
        loss = 0

        for metric in self.all_metrics:
            if metric in self.metric_trgs:
                trg = self.metric_trgs[metric]
            # elif metric in self.static_metrics or metric in self.ctrl_loss_metrics:
            else:
                trg = self.static_trgs[metric]
            # else:
            #     raise Exception("Metric should have static target.")
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

    def get_static_loss(self):
        loss = 0

        for metric in self.static_metric_names:
            # We already pop these from static_metrics during init
            # if metric in self.ctrl_metrics:
            #     continue
            trg = self.static_trgs[metric]
            val = self.metrics[metric]

            if isinstance(trg, tuple):
                loss_m = -abs(np.arange(*trg) - val).min()
            else:
                loss_m = -abs(trg - val)
            loss_m = loss_m * self.metric_weights[metric]
            loss += loss_m

        return loss

    def get_reward(self):
        reward = 0
        # for k in self.all_metrics:
        #    if k in self.ctrl_metrics:
        #        trg = self.cond_trgs[k]
        #    elif k in self.static_trgs:
        #        trg = self.static_trgs[k]
        #    else:
        #        raise Exception("Invalid metric")
        #    if not isinstance(trg, tuple):
        #        low = trg
        #        high = trg
        #    else:
        #        low, high = trg
        #    reward += get_range_reward(self.metrics[k], self.last_metrics[k], low, high) * self.unwrapped._prob._reward_weights[k]
        #       print(self.metrics)
        #       print(reward)
        # return reward
        #           reward = loss

        if not self.SC_RCT:
            loss = self.get_loss()
        else:
            # FIXME: why do we do this?
            loss = self.get_ctrl_loss()

        # max_loss is positive, loss is negative. Normalize reward between 0 and 1.
        # reward = (self.max_loss + loss) / (self.max_loss)

        # return reward

        reward = loss - self.last_loss
        self.last_loss = loss

        return reward

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
        if self.render_gui and self.conditional:
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
#    def reset(self):
##       self.noise = OpenSimplex()
#        return self.env.reset()


class UniformNoiseyTargets(gym.Wrapper):
    """A bunch of simplex noise instances modulate target metrics."""

    def __init__(self, env, **kwargs):
        super(UniformNoiseyTargets, self).__init__(env)
        self.cond_bounds = self.env.unwrapped.cond_bounds
        self.num_params = self.num_params
        self.midep_trgs = kwargs.get("midep_trgs", False)

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

    def reset(self):
        self.set_rand_trgs()

        return self.env.reset()


class ALPGMMTeacher(gym.Wrapper):
    def __init__(self, env, **kwargs):

        from teachDRL.teachers.algos.alp_gmm import ALPGMM

        super(ALPGMMTeacher, self).__init__(env)
        self.cond_bounds = self.env.unwrapped.cond_bounds
        self.midep_trgs = False
        env_param_lw_bounds = [self.cond_bounds[k][0] for k in self.ctrl_metrics]
        env_param_hi_bounds = [self.cond_bounds[k][1] for k in self.ctrl_metrics]
        self.alp_gmm = ALPGMM(env_param_lw_bounds, env_param_hi_bounds)
        self.trg_vec = None
        self.trial_reward = 0
        self.n_trial_steps = 0

    def reset(self):
        if self.trg_vec is not None:
            if self.n_trial_steps == 0:
                # This is some whackness that happens when we reset manually from the inference script.
                rew = 0
            else:
                rew = self.trial_reward / self.n_trial_steps
            self.alp_gmm.update(self.trg_vec, rew)
        trg_vec = self.alp_gmm.sample_task()
        self.trg_vec = trg_vec
        trgs = {k: trg_vec[i] for (i, k) in enumerate(self.ctrl_metrics)}
        #       print(trgs)
        self.set_trgs(trgs)
        self.trial_reward = 0
        self.n_trial_steps = 0

        return self.env.reset()

    def step(self, action, **kwargs):
        obs, rew, done, info = self.env.step(action, **kwargs)
        self.trial_reward += rew
        self.n_trial_steps += 1

        return obs, rew, done, info
