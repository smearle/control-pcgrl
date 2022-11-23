"""
Run a trained agent for qualitative analysis.
"""
import json
import os
import pickle
from pdb import set_trace as TT

import cv2
import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from numpy import asarray
from PIL import Image

from control_pcgrl.rl.args import get_args, parse_pcgrl_args, prob_cond_metrics
from control_pcgrl.rl.envs import make_vec_envs
from control_pcgrl.rl.utils import (
    get_action,
    get_crop_size,
    get_map_width,
    get_env_name,
    get_exp_name,
    load_model,
    PROB_CONTROLS,
#   max_exp_idx,
)

DPI = 100

# For 1D data, do we use a bar chart instead of a heatmap?
BAR_CHART = False

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2
plt.rcParams.update({'font.size': 13})


def div_calc(tokens):
    div_score = np.sum([np.sum(a != b) for a in tokens for b in tokens]) / (
        len(tokens) * (len(tokens) - 1)
    )
    div_score = div_score / (tokens[0].shape[0] * (tokens[0].shape[1]))

    return div_score


def evaluate(game, representation, infer_kwargs, fix_trgs=False, **kwargs):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    global N_BINS
    global N_MAPS
    global N_TRIALS

    infer_kwargs = {**infer_kwargs, "inference": True, "evaluate": True}
    #   max_trials = kwargs.get("max_trials", -1)
#   n = kwargs.get("n", None)
    exp_id = infer_kwargs.get('exp_id')
    #   map_width = infer_kwargs.get("map_width")
    max_steps = infer_kwargs.get("max_step")
    eval_controls = infer_kwargs.get("eval_controls")
    env_name = get_env_name(game, representation)
    #   env_name = '{}-{}-v0'.format(game, representation)
    exp_name = get_exp_name(game, representation, **kwargs)
    levels_im_name = "{}_{}-bins_levels.png"

#   if n is None:
#       if EXPERIMENT_ID is None:
#           n = max_exp_idx(exp_name)
#           print(
#               "Experiment index not specified, setting index automatically to {}".format(
#                   n
#               )
#           )
#       else:
#           n = EXPERIMENT_ID

#   if n == 0:
#       raise Exception(
#           "Did not find ranked saved model of experiment: {}".format(exp_name)
#       )
    crop_shape = infer_kwargs.get("crop_shape")

    # if crop_size == -1:
    #     infer_kwargs["crop_size"] = get_crop_size(game)
    log_dir = os.path.join(EXPERIMENT_DIR, '{}_{}_log'.format(exp_name, exp_id))
    eval_dir = os.path.join(log_dir, 'eval')
    if not os.path.isdir(eval_dir):
        os.mkdir(eval_dir)
#   log_dir = "{}/{}_{}_log".format(EXPERIMENT_DIR, exp_name, exp_id)
    data_path = os.path.join(eval_dir, "{}_{}_eval_data".format(N_BINS, eval_controls))
    data_path_levels = os.path.join(eval_dir, "{}_eval_data_levels".format(N_BINS))
    if fix_trgs:
        data_path += "_fixTrgs"
        data_path_levels += "_fixTrgs"
    data_path += ".pkl"
    data_path_levels += ".pkl"

    if VIS_ONLY:
#       if RENDER_LEVELS:
#           eval_data_levels = pickle.load(open(data_path_levels, "rb"))
#           eval_data_levels.render_levels()

#           return
        eval_data = pickle.load(open(data_path, "rb"))
        # FIXME: just for backward compatibility
        eval_data.eval_dir = eval_dir
        eval_data.visualize_data(eval_dir, fix_trgs)
        eval_data.render_levels()
#       eval_data.hamming_heatmap(None, eval_data.div_scores)

        return
    # no log dir, 1 parallel environment
    n_cpu = infer_kwargs.get("n_cpu")
    if 'path-length' in eval_controls or not eval_controls:
        infer_kwargs['render_path'] = True
    env, dummy_action_space, n_tools = make_vec_envs(
        env_name, representation, None, **infer_kwargs
    )
    model = load_model(
        log_dir, load_best=infer_kwargs.get("load_best"), n_tools=n_tools
    )
    #   model.set_env(env)
    env.action_space = dummy_action_space
    env = env.envs[0]
    # Record final values of each trial
    #   if 'binary' in env_name:
    #       path_lengths = []
    #       changes = []
    #       regions = []
    #       infer_info = {
    #           'path_lengths': [],
    #           'changes': [],
    #           'regions': [],
    #           }

    if n_cpu == 1:
#       control_bounds = env.envs[0].get_control_bounds()
        # control_bounds = env.get_control_bounds()
        control_bounds = env.cond_bounds
    elif n_cpu > 1:
        raise Exception("no homie, no")
        # supply opts and kwargs
        env.remotes[0].send(("env_method", ("get_control_bounds", [], {})))
        control_bounds = env.remotes[0].recv()

    if not eval_controls:
        eval_controls = control_bounds.keys()
    if len(control_bounds) == 0:
        # Then this is a non-controllable agent.
        # Can't we just do this in all cases though?
        control_bounds = env.cond_bounds
    ctrl_bounds = []
    for k in eval_controls:
        bounds = control_bounds[k]
        if 'path-length' in k:
            if 'zelda' in game:
                bounds = (3, bounds[1])
            else:
                bounds = (1, bounds[1])
        if 'nearest-enemy' in k:
            bounds = (1, bounds[1])
        if 'sol-length' in k:
            bounds = (1, bounds[1])
        if 'regions' in k:
            bounds = (1, bounds[1])
        ctrl_bounds.append((k, bounds))

    #   if len(ctrl_bounds) == 0 and DIVERSITY_EVAL:
    #       N_MAPS = 100
    #       N_TRIALS = 1
    # Hackish get initial states
    init_states = []

    for i in range(N_MAPS):
        env.reset()
        # TODO: set initial states in either of these domains?

        if not (RCT or SC):
            init_states.append(env.unwrapped._rep._map)
    N_EVALS = N_TRIALS * N_MAPS
    levels_x_labels = []
    levels_y_labels = []

    def eval_static_trgs():
        '''Run an evaluation on the default values for all level metrics. For both controllable and vanilla agents. 
        The latter's "home turf."'''
        N_BINS = None
        level_images = []
        cell_scores = np.zeros(shape=(1, 1, N_EVALS))
        div_scores = np.zeros(shape=(1, 1))
        cell_static_scores = np.zeros(shape=(1, 1, N_EVALS))
        cell_ctrl_scores = np.zeros(shape=(1, 1, N_EVALS))
        level_tokens = None

#       if DIVERSITY_EVAL:
#           n_row = 1
#           n_col = 1
#       else:
        n_row = 2
        n_col = 5

        for i in range(n_row):
            level_images_y = []

            for j in range(n_col):
                net_score, ctrl_score, static_score, level_image, tokens = eval_episodes(
                    model,
                    env,
                    N_EVALS,
                    n_cpu,
                    init_states,
                    eval_dir,
                    env.unwrapped._prob.static_trgs,
                    max_steps,
                )
                level_images_y.append(level_image)
                cell_scores[0, 0, :] = net_score
                div_score = np.sum(
                    [np.sum(a != b) for a in tokens for b in tokens]
                ) / (len(tokens) * (len(tokens) - 1))
                div_score = div_score / (map_width * map_width)
                div_scores[0, 0] = div_score

            level_images.append(np.hstack(level_images_y))

        image = np.vstack(level_images[::-1])
        image = Image.fromarray(image)
        image.save(
            os.path.join(eval_dir, levels_im_name.format(ctrl_names, N_BINS))
        )

        return cell_scores, cell_static_scores, cell_ctrl_scores, div_scores, level_tokens, image

    if len(ctrl_bounds) == 0:
        # If we didn't train with controls, we'll evaluate inside a grid of targets (on the controllable agents' turf)
        # and record scores for the cell corresponding to the default static targets (on the vanilla turf),
        # depending on the value of fix_trgs.
        ctrl_names = prob_cond_metrics[problem]
        ctrl_bounds = [(k, env.cond_bounds[k]) for k in ctrl_names]

    if fix_trgs:
        ctrl_names = None
        ctrl_ranges = None
        cell_scores, cell_static_scores, cell_ctrl_scores, div_scores, level_tokens, image = eval_static_trgs()

    elif len(ctrl_bounds) == 1:
        ctrl_name = ctrl_bounds[0][0]
        bounds = ctrl_bounds[0][1]
        print(ctrl_bounds)
        step_size = max((bounds[1] - bounds[0]) / (N_BINS[0] - 1), 1)
        eval_trgs = np.arange(bounds[0], bounds[1] + 1, step_size)
        level_images = []
        cell_scores = np.zeros((len(eval_trgs), 1, N_EVALS))
        cell_ctrl_scores = np.zeros(shape=(len(eval_trgs), 1, N_EVALS))
        cell_static_scores = np.zeros(shape=(len(eval_trgs), 1, N_EVALS))
        level_tokens = []
        div_scores = np.zeros((len(eval_trgs), 1))

        for i, trg in enumerate(eval_trgs):
            trg_dict = {ctrl_name: trg}
            print("evaluating control targets: {}".format(trg_dict))
            #           set_ctrl_trgs(env, {ctrl_name: trg})
            net_score, ctrl_score, static_score, level_image, tokens = eval_episodes(
                model, env, N_EVALS, n_cpu, init_states, eval_dir, trg_dict, max_steps
            )
            div_score = div_calc(tokens)
            div_scores[i, 0] = div_score
            if i % LVL_RENDER_INTERVAL == 0:
                level_images.append(level_image)
                levels_x_labels.append(trg)
            cell_scores[i, :, :] = net_score
            cell_ctrl_scores[i, :, :] = ctrl_score
            cell_static_scores[i, :, :] = static_score
            level_tokens.append(tokens)
        ctrl_names = (ctrl_name, None)
        ctrl_ranges = (eval_trgs, None)
        #       if "regions" in ctrl_ranges:
        #           # hack it to ensure our default static trgs are in the heatmap, so we can compare on baseline's turf
        #           ctrl_ranges["regions"][0] = 1

        ims = np.hstack(level_images)
        image = Image.fromarray(ims)
        # image.save(os.path.join(eval_dir, levels_im_name.format(ctrl_names, N_BINS)))

    elif len(ctrl_bounds) >= 2:
        ctrl_0, ctrl_1 = ctrl_bounds[0][0], ctrl_bounds[1][0]
        b0, b1 = ctrl_bounds[0][1], ctrl_bounds[1][1]
        step_0 = max((b0[1] - b0[0]) / (N_BINS[0] - 1), 1)
        # step_0 = (b0[1] - b0[0]) / (N_BINS[0] - 1)
        step_1 = max((b1[1] - b1[0]) / (N_BINS[-1] - 1), 1)
        # step_1 = (b1[1] - b1[0]) / (N_BINS[-1] - 1)
        trgs_0 = np.arange(b0[0], b0[1] + 0.5, step_0)
        trgs_1 = np.arange(b1[0], b1[1] + 0.5, step_1)
        cell_scores = np.zeros(shape=(len(trgs_0), len(trgs_1), N_EVALS))
        div_scores = np.zeros(shape=(len(trgs_0), len(trgs_1)))
        cell_ctrl_scores = np.zeros(shape=(len(trgs_0), len(trgs_1), N_EVALS))
        cell_static_scores = np.zeros(shape=(len(trgs_0), len(trgs_1), N_EVALS))
        level_tokens = [[None] * len(trgs_0)] * len(trgs_1)  # Wait what?
        trg_dict = env.static_trgs
        trg_dict = dict(
            [
                (k, min(v)) if isinstance(v, tuple) else (k, v)
                for (k, v) in trg_dict.items()
            ]
        )
        level_images = []

        for i, t0 in enumerate(trgs_0):
            level_images_y = []

            for j, t1 in enumerate(trgs_1):
                ctrl_trg_dict = {ctrl_0: t0, ctrl_1: t1}
                trg_dict.update(ctrl_trg_dict)
                print("evaluating control targets: {}".format(trg_dict))
                #           set_ctrl_trgs(env, {ctrl_name: trg})
                net_score, ctrl_score, static_score, level_image, tokens = eval_episodes(
                    model,
                    env,
                    N_EVALS,
                    n_cpu,
                    init_states,
                    eval_dir,
                    trg_dict,
                    max_steps,
                )

                if j % LVL_RENDER_INTERVAL == 0:
                    level_images_y.append(level_image)
                    levels_y_labels.append(t1)
                cell_scores[i, j, :] = net_score
                cell_ctrl_scores[i, j, :] = ctrl_score
                cell_static_scores[i, j, :] = static_score
                div_score = div_calc(tokens)
                div_scores[i, j] = div_score
            #               level_tokens[j][i] = tokens

            if i % LVL_RENDER_INTERVAL == 0:
                level_images.append(np.hstack(level_images_y))
                levels_x_labels.append(t0)

        #           level_tokens.append(tokens)
        ctrl_names = (ctrl_0, ctrl_1)
        ctrl_ranges = (trgs_0, trgs_1)
        image = None

        image = np.vstack(level_images[::-1])
        image = Image.fromarray(image)
        image.save(os.path.join(eval_dir, levels_im_name.format(ctrl_names, N_BINS)))
    levels_im_path = os.path.join(eval_dir, levels_im_name.format(ctrl_names, N_BINS))

    eval_data = EvalData(
        ctrl_names,
        env.static_metrics,
        ctrl_ranges,
        cell_scores,
        cell_ctrl_scores,
        cell_static_scores,
        div_scores=div_scores,
        eval_dir=eval_dir,
        levels_image=image,
        levels_im_path=levels_im_path,
        levels_x_labels=levels_x_labels,
        levels_y_labels=levels_y_labels,
    )
    pickle.dump(eval_data, open(data_path, "wb"))
    eval_data.visualize_data(eval_dir, fix_trgs)

#   else:
#       levels_im_path = os.path.join(
#           eval_dir, levels_im_name.format(ctrl_names, N_BINS)
#       )
#       eval_data_levels = EvalData(
#           ctrl_names,
#           ctrl_ranges,
#           cell_scores,
#           cell_ctrl_scores,
#           cell_static_scores,
#           div_scores=div_scores,
#           eval_dir=eval_dir,
#           levels_image=image,
#           levels_im_path=levels_im_path,
#       )
#       pickle.dump(eval_data_levels, open(data_path_levels, "wb"))
    if not fix_trgs:
        eval_data.render_levels()

    if DIVERSITY_EVAL:
#       eval_data = eval_data

        if fix_trgs:
            eval_data.save_stats(div_scores=div_scores, fix_trgs=fix_trgs)
        else:
            pass
#           eval_data.hamming_heatmap(level_tokens, div_scores=div_scores)

    env.close()


def eval_episodes(
    model, env, n_trials, n_envs, init_states, eval_dir, trg_dict, max_steps
):
    env.set_trgs(trg_dict)
    eval_scores = np.zeros(n_trials)
    eval_ctrl_scores = np.zeros(n_trials)
    eval_static_scores = np.zeros(n_trials)
    n = 0
    # FIXME: why do we need this?
    tokens = []

    max_rew = -np.inf
    while n < n_trials:
        if not (RCT or SC):
            env.set_map(init_states[n % N_MAPS])
        elif SC:
            # Resize gui window for simcity
            env.win1.editMapView.changeScale(0.77)
            env.win1.editMapView.centerOnTile(20, 16)
        obs = env.reset()
        #       epi_reward_weights = np.zeros((max_step, n_envs))
        i = 0
        # note that this is weighted loss
        init_loss = env.get_loss()
        init_ctrl_loss = env.get_ctrl_loss()
        init_static_loss = env.get_static_loss()
        #       print('initial loss, net: {}, static: {}, ctrl: {}'.format(init_loss, init_static_loss, init_ctrl_loss))
        done = False

        while not done:
            #           if i == max_steps - 1:

            action, _ = model.predict(obs)
            obs, rewards, done, info = env.step(action)
#           if done: 
#               pass
            i += 1
        # print('total episode steps:', i)
        final_loss = env.get_loss()
        final_ctrl_loss = env.get_ctrl_loss()
        final_static_loss = env.get_static_loss()
        if RENDER_LEVELS and final_loss > max_rew:
            max_rew = final_loss
            image = env.render("rgb_array")

            if SC and i == max_steps - 1:
                # FIXME lmao fucking stupid as all heck (wait is this the right directory? No)
                im_path = os.path.join(eval_dir, "{}_level.png".format(trg_dict))
                image = env.win1.editMapView.buffer.write_to_png(
                    im_path
                )
                em = env.win1.editMapView
                image = Image.open(im_path)
                image = np.array(image)
                print(image.shape)
                image = image[:, 400:-400, :]
                print(image.shape)

            if RCT and i == max_steps - 1:
                image = Image.fromarray(image.transpose(1, 0, 2))

        # Ayo wtf is this garbage?
        if not (SC or RCT):
            curr_tokens = env.unwrapped._rep._map
        elif SC:
            curr_tokens = env.state.argmax(axis=0)
        elif RCT:
            curr_tokens = env.rct_env.park.map[0]
    #           epi_reward_weights[i] = rewards



        #       print('final loss, net: {}, static: {}, ctrl: {}'.format(final_loss, final_static_loss, final_ctrl_loss))
        # what percentage of loss (distance from target) was recovered?
        eps = 0.001
        max_loss = max(abs(init_loss), eps)
        max_ctrl_loss = max(abs(init_ctrl_loss), eps)
        max_static_loss = max(abs(init_static_loss), eps)
        score = (final_loss - init_loss) / abs(max_loss)
        ctrl_score = (final_ctrl_loss - init_ctrl_loss) / abs(max_ctrl_loss)
        static_score = (final_static_loss - init_static_loss) / abs(max_static_loss)
        eval_scores[n] = score
        eval_ctrl_scores[n] = ctrl_score
        eval_static_scores[n] = static_score
        n += n_envs
        tokens.append(curr_tokens)

    eval_score = eval_scores.mean()
    eval_ctrl_score = eval_ctrl_scores.mean()
    eval_static_score = eval_static_scores.mean()
    print("eval score: {}".format(eval_score))
    print("control score: {}".format(eval_ctrl_score))
    print("static score: {}".format(eval_static_score))

    if RENDER_LEVELS and not SC:
        # we hackishly save it for SC up above already
#       image.save(os.path.join(eval_dir, "{}_level.png".format(trg_dict)))
        level_image = asarray(image)
    else:
        level_image = None

    return eval_scores, eval_ctrl_scores, eval_static_scores, level_image, tokens


class EvalData:
    def __init__(
        self,
        ctrl_names,
        static_names,
        ctrl_ranges,
        cell_scores,
        cell_ctrl_scores,
        cell_static_scores,
        div_scores,
        eval_dir,
        levels_image=None,
        levels_im_path=None,
        levels_x_labels=None,
        levels_y_labels=None,
    ):
        self.ctrl_names = ctrl_names
        self.static_names = static_names
        self.ctrl_ranges = ctrl_ranges
        self.cell_scores = cell_scores
        self.cell_ctrl_scores = cell_ctrl_scores
        self.cell_static_scores = cell_static_scores
        self.div_scores = div_scores
        self.levels_image = levels_image
        self.levels_im_path = levels_im_path
        self.eval_dir = eval_dir
        self.levels_x_labels = levels_x_labels
        self.levels_y_labels = levels_y_labels
        if self.ctrl_names:
            self.ctrl_names = list(self.ctrl_names)
            for i, cn in enumerate(self.ctrl_names):
                if cn == 'sol-length':
                    self.ctrl_names[i] = 'solution-length'

    def visualize_data(self, eval_dir, fix_trgs):
        self.save_stats(div_scores=self.div_scores, fix_trgs=fix_trgs)

        if fix_trgs:
            return

        def create_heatmap(title, cbar_label, data, vrange=(0,100), cmap_name=None):
            data = data * 100
            data = np.clip(data, -100, 100)
            if vrange is None:
                vmin, vmax = data.min(), data.max()
            else:
                vmin, vmax = vrange
            fig, ax = plt.subplots(dpi=100)
            # percentages from ratios
            #           data = data.T

            if data.shape[1] == 1:
                plt.xlabel(ctrl_names[0])
                data = data.T
                fig.set_size_inches(10, 2)
                tick_idxs = np.arange(
                    0, cell_scores.shape[0], max(1, (cell_scores.shape[0] // 10))
                )
                ticks = np.arange(cell_scores.shape[0])
                ticks = ticks[tick_idxs]
                ax.set_xticks(ticks)
                labels = np.array(
                    [int(round(x, 0)) for (i, x) in enumerate(ctrl_ranges[0])]
                )
                labels = labels[tick_idxs]
                ax.set_xticklabels(labels)

                if BAR_CHART:
                    #                   low = data[0].min()
                    low = 0
                    high = 100
                    plt.ylim([low, high])
                    ax.bar(ticks, data[0])
                else:
                    ax.set_yticks([])
            else:
                data = data[::-1, :]
                ax.set_xticks(np.arange(cell_scores.shape[1]))
                ax.set_yticks(np.arange(cell_scores.shape[0]))
                ax.set_xticklabels([int(round(x, 0)) for x in ctrl_ranges[1]])
                ax.set_yticklabels([int(round(x, 0)) for x in ctrl_ranges[0][::-1]])
            # Create the heatmap
            #           im = ax.imshow(data, aspect='auto', vmin=-100, vmax=100)

            if not BAR_CHART or data.shape[1] != 1:

                # Create colorbar
                if cmap_name:
                    cmap = plt.get_cmap(cmap_name)
                else:
                    cmap = None
#                   cmap = colors.ListedColormap(["b", "r", "y", "r"])
                im = ax.imshow(data, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
                cbar = ax.figure.colorbar(im, ax=ax)
                # cbar.ax.set_ylabel("", rotation=90, va="bottom")

                # We want to show all ticks...
                if data.shape[0] != 1:
                    # ... and label them with the respective list entries
                    plt.xlabel(ctrl_names[1])
                    plt.ylabel(ctrl_names[0])

            cbar.set_label(cbar_label, labelpad=6)
            # Rotate the tick labels and set their alignment.
            plt.setp(
                ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
            )

            ax.set_title(title)
            fig.tight_layout()

            plt.savefig(
                os.path.join(
                    eval_dir, "{}_{}.svg".format(ctrl_names, title.replace("%", ""))
                )
            )

        #           plt.show()

        ctrl_names = self.ctrl_names
        ctrl_ranges = self.ctrl_ranges
        cell_scores = self.cell_scores
        cell_ctrl_scores = self.cell_ctrl_scores
        cell_static_scores = self.cell_static_scores

        title = "All goals"
        cbar_label = "progress (%)"
        create_heatmap(title, cbar_label, cell_scores.mean(-1))

        title = "Controlled goals"
        create_heatmap(title, cbar_label, cell_ctrl_scores.mean(-1))

#       title = "Fixed goals: {}".format(', '.join(self.static_names))
        title = "Fixed goals"
        create_heatmap(title, cbar_label, cell_static_scores.mean(-1))

        title = "Diversity"
        if self.div_scores.shape[1] == 1:
            cbar_label = "per-tile \ndifference (%)"
        else:
            cbar_label = "per-tile difference (%)"
        create_heatmap(title, cbar_label, self.div_scores, vrange=None, cmap_name="inferno")

    def pairwise_hamming(self, a, b):
        return np.sum(a != b)

    def save_stats(self, div_scores=np.zeros(shape=(1, 1)), fix_trgs=False):
        def get_stat_subdict(stats):
            # Prevent explosive negative scores from initializing close to targets
            stats = np.clip(stats, 0, stats.max())
            return {"mean": stats.mean(), "std_dev": stats.std()}

        scores = {
            "net_score": get_stat_subdict(self.cell_scores),
            "ctrl_score": get_stat_subdict(self.cell_ctrl_scores),
            "fixed_score": get_stat_subdict(self.cell_static_scores),
            "diversity_score": get_stat_subdict(div_scores),
            "n_maps_per_cell": N_MAPS,
            "n_trials_per_map": N_TRIALS,
            "controls": self.ctrl_names,
        }

        if fix_trgs:
            filename = "scores_fixTrgs.json"
        else:
            filename = "scores_{}_ctrlTrgs.json".format(self.ctrl_names)
        with open(os.path.join(self.eval_dir, filename), "w") as fp:
            json.dump(scores, fp, ensure_ascii=False, indent=4)

    def render_levels(self):
        ctrl_names = self.ctrl_names
        if 'sokoban_ctrl' in self.eval_dir:
            plt.rcParams.update({'font.size': 10})
        else:
            plt.rcParams.update({'font.size': 22})
        if ctrl_names[1] is not None:
            fig, ax = plt.subplots()
            fig.set_figwidth(np.array(self.levels_image).shape[0] / DPI)
            fig.set_figheight(np.array(self.levels_image).shape[1] / DPI)
        else:
            fig, ax = plt.subplots()
            fig.set_figwidth(np.array(self.levels_image).shape[1] / DPI)
            fig.set_figheight(np.array(self.levels_image).shape[0] / DPI)
        ax.imshow(self.levels_image)
        #       ax.axis["xzero"].set_axisline_style("-|>")
        # plt.tick_params(
        #    axis='x',
        #    which='both',
        #    bottom=False,
        #    top=False,
        #    labelbottom=False)

        if ctrl_names[1] is None:
            plt.xlabel(ctrl_names[0])
            # wHut???
            im_width = np.array(self.levels_image).shape[1] / self.cell_scores.shape[0]
            plt.xticks(
                (np.arange(N_LVL_BINS) * im_width + im_width / 2) * (N_BINS[-1] / N_LVL_BINS),
                # np.arange(N_LVL_BINS) * (im_width * LVL_RENDER_INTERVAL) + im_width / 2,
                labels=[int(round(self.ctrl_ranges[0][i * LVL_RENDER_INTERVAL], 0)) for i in range(N_LVL_BINS)],
            )
            pad_inches = 0
            hspace = 0
            bottom = 0
            plt.tick_params(
                axis="y", which="both", left=False, right=False, labelleft=False
            )
        else:
            plt.xlabel(ctrl_names[1])
            plt.ylabel(ctrl_names[0])
            im_width = np.array(self.levels_image).shape[1] / self.cell_scores.shape[1]
            im_height = np.array(self.levels_image).shape[0] / self.cell_scores.shape[0]
            n_x_lvls = len(self.levels_y_labels)
            plt.xticks(
                (np.arange(n_x_lvls) * im_width + im_width / 2) * (N_BINS[-1] / N_LVL_BINS),
                labels=[int(round(i, 0)) for i in self.levels_y_labels]
            )
            n_y_lvls = len(self.levels_x_labels)
            plt.yticks(
                (np.arange(n_y_lvls) * im_height + im_height / 2) * (N_BINS[-1] / N_LVL_BINS),
                labels=[int(round(i, 0)) for i in self.levels_x_labels][::-1],
            )
            #           ax.set_xticklabels([round(x, 1) for x in ctrl_ranges[0]])
            #           ax.set_yticklabels([round(x, 1) for x in ctrl_ranges[1][::-1]])

            pad_inches = 0
            hspace = 0
            bottom = 0.05

        # plt.gca().set_axis_off()
        plt.subplots_adjust(
            top=1, bottom=bottom, right=1, left=0, hspace=hspace, wspace=0
        )
        plt.margins(0, 0)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.savefig(self.levels_im_path, bbox_inches="tight", pad_inches=pad_inches, dpi=DPI)
        plt.savefig(self.levels_im_path.replace('.png', '.svg'), bbox_inches="tight", pad_inches=pad_inches, format='svg', dpi=DPI)
        # plt.show()
        plt.rcParams.update({'font.size': 13})


# NOTE: let's not try multiproc how about that :~)
# def eval_episodes(model, env, n_trials, n_envs):
#    eval_scores = np.zeros(n_trials)
#    n = 0
#    # FIXME: why do we need this?
#    env.reset()
#    while n < n_trials:
#
#        obs = env.reset()
##       epi_reward_weights = np.zeros((max_step, n_envs))
#        i = 0
# env.remotes[0].send(('env_method', ('get_metric_vals', [], {})))  # supply opts and kwargs
##       init_metric_vals = env.remotes[0].recv()
#        [remote.send(('env_method', ('get_loss', [], {}))) for remote in env.remotes]
#        # note that this is weighted loss
#        init_loss = np.sum([remote.recv() for remote in env.remotes])
#        dones = np.array([False])
#        while not dones.all():
#            action, _ = model.predict(obs)
#            obs, rewards, dones, info = env.step(action)
##           epi_reward_weights[i] = rewards
#            i += 1
#        # since reward is weighted loss
#        final_loss = np.sum(rewards)
#        # what percentage of loss (distance from target) was recovered?
#        score = (final_loss - init_loss) / abs(init_loss)
# env.remotes[0].send(('env_method', ('get_metric_vals', [], {})))  # supply opts and kwargs
##       final_metric_vals = env.remotes[0].recv()
#        eval_scores[n] = score
#        n += n_envs #    return eval_scores.mean()
#
# def set_ctrl_trgs(env, trg_dict):
#    [remote.send(('env_method', ('set_trgs', [trg_dict], {}))) for remote in env.remotes]
args = get_args()
args.add_argument(
    "--vis_only",
    help="Just load data from previous evaluation and visualize it.",
    action="store_true",
)
args.add_argument(
    "--eval_controls",
    help="Which controls to evaluate and visualize.",
    nargs="+",
    default=[],
)
args.add_argument(
    "--n_maps",
    help="Number maps on which to simulate in each cell.",
    default=10,
    type=int,
)
args.add_argument(
    "--n_trials",
    help="Number trials for which to simulate on each map.",
    default=1,
    type=int,
)
# opts.add_argument('--step_size',
#        help='Bin size along either dimension.',
#        default=20,
#        type=int,
#        )
args.add_argument(
    "--n_bins",
    help="How many bins along each dimension (a sequence of ints).",
    nargs="+",
    type=int,
    default=(10, 10),
)
args.add_argument(
    "--render_levels",
    help="Save final maps (default to only one eval per cell)",
    action="store_true",
)
# opts.add_argument('--diversity_eval',
#        help='Evaluate average pairwise hamming distance per cell',
#        action='store_true',
#        )
opts = parse_pcgrl_args(args)
global VIS_ONLY
VIS_ONLY = opts.vis_only
# DIVERSITY_EVAL = opts.diversity_eval

# For locating trained model
global EXPERIMENT_ID
global EXPERIMENT_DIR
# EXPERIMENT_DIR = 'hpc_runs/runs'

if not opts.HPC:
    EXPERIMENT_DIR = "../rl_runs"
else:
    EXPERIMENT_DIR = "hpc_runs"
EXPERIMENT_ID = opts.exp_id
problem = opts.problem

if "RCT" in problem:
    RCT = True
else:
    RCT = False

if "Micropolis" in problem:
    SC = True
else:
    SC = False
representation = opts.representation
cond_metrics = opts.controls
conditional = len(cond_metrics) > 0
midep_trgs = opts.midep_trgs
ca_action = opts.ca_action
alp_gmm = opts.alp_gmm
train_change_percentage = opts.change_percentage
# Ignore change percentage, we care only about reaching targets or hitting a max number of steps
infer_change_percentage = 1
# TODO: properly separate these kwarg dictionaries, so that one is for loading (specifying training run 
# hyperparameters), and the other is for inference (what settings to evaluate with)
if conditional:
    max_step = opts.max_step
    #   if max_step is None:
    #       if RENDER_LEVELS:
    #           max_step = 10000
    #       else:
    #           max_step = 5000

    if ca_action:
        max_step = 50
else:
    max_step = None

max_step = 1000


kwargs = {
    "change_percentage": train_change_percentage,
    "conditional": conditional,
    "cond_metrics": cond_metrics,
    "alp_gmm": alp_gmm,
    "max_step": max_step,
    # 'target_path': 105,
    # 'n': 4, # rank of saved experiment (by default, n is max possible)
}

#RENDER_LEVELS = opts.render_levels
RENDER_LEVELS = True
DIVERSITY_EVAL = True

map_width = get_map_width(problem)
#if problem == "sokobangoal":
#    map_width = 5
#else:
#    map_width = 16

# For inference
infer_kwargs = {
    "change_percentage": infer_change_percentage,
    # 'target_path': 200,
    "conditional": conditional,
    "cond_metrics": cond_metrics,
    "max_step": max_step,
    "render": opts.render,
    # TODO: multiprocessing
    #       'n_cpu': opts.n_cpu,
    "n_cpu": 1,
    "load_best": opts.load_best,
    "midep_trgs": midep_trgs,
    "infer": True,
    "ca_action": ca_action,
    "map_width": map_width,
    "eval_controls": opts.eval_controls,
    "crop_size": opts.crop_size,
    "exp_id": opts.exp_id,
}

global N_BINS
N_BINS = tuple(opts.n_bins)
N_LVL_BINS = 4
assert N_BINS[0] % (N_LVL_BINS - 1) == 1
LVL_RENDER_INTERVAL = N_BINS[0] // (N_LVL_BINS - 1)

N_TRIALS = 1

N_MAPS = opts.n_maps
#   N_TRIALS = opts.n_trials

if __name__ == "__main__":

    # Evaluate controllability
    # Evaluate fixed quality of levels, or controls at default targets
#   if not VIS_ONLY:
#       evaluate(problem, representation, infer_kwargs, fix_trgs=True, **kwargs)
#   if not conditional:
    control_sets = PROB_CONTROLS[problem]
    for i, eval_ctrls in enumerate(control_sets):
        # Then evaluate over some default controls (otherwise use those that we trained on)
        # TODO: for each experiment, repeat for a set of control-sets
#       cond_metrics = set(infer_kwargs.get('cond_metrics'))
#       cond_metrics.update(set(eval_ctrls))
#       cond_metrics = [e for e in cond_metrics]
        infer_kwargs.update({'eval_controls': eval_ctrls, 'cond_metrics': cond_metrics})
        evaluate(problem, representation, infer_kwargs, fix_trgs=False, **kwargs)
#   else:
#           evaluate(problem, representation, infer_kwargs, fix_trgs=False, **kwargs)
#   evaluate(test_params, game, representation, experiment, infer_kwargs, **kwargs)
#   analyze()
