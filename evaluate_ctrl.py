
"""
Run a trained agent for qualitative analysis.
"""
import os
from PIL import Image
from pdb import set_trace as T
import numpy as np
from numpy import asarray
import cv2
from utils import get_exp_name, max_exp_idx, load_model, get_action, get_crop_size, get_env_name
from envs import make_vec_envs
from matplotlib import pyplot as plt
import pickle
import json
from matplotlib import colors

# For 1D data, do we use a bar chart instead of a heatmap?
BAR_CHART = True

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

def div_calc(tokens):
    div_score = np.sum([np.sum(a != b) for a in tokens for b in tokens]) / (len(tokens) * (len(tokens) - 1))
    div_score = div_score / (tokens[0].shape[0] * (tokens[0].shape[1]))
    return div_score


def evaluate(game, representation, infer_kwargs, **kwargs):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    global N_BINS
    global N_MAPS
    global N_TRIALS

    infer_kwargs = {
            **infer_kwargs,
            'inference': True,
            'evaluate': True
            }
    max_trials = kwargs.get('max_trials', -1)
    n = kwargs.get('n', None)
    map_width = infer_kwargs.get('map_width')
    max_steps = infer_kwargs.get('max_step')
    eval_controls = infer_kwargs.get('eval_controls')
    env_name = get_env_name(game, representation)
#   env_name = '{}-{}-v0'.format(game, representation)
    exp_name = get_exp_name(game, representation, **kwargs)
    levels_im_name = "{}_{}-bins_levels.png"
    if n is None:
        if EXPERIMENT_ID is None:
            n = max_exp_idx(exp_name)
        else:
            n = EXPERIMENT_ID
    if n == 0:
        raise Exception('Did not find ranked saved model of experiment: {}'.format(exp_name))
    crop_size = infer_kwargs.get('cropped_size')
    if crop_size == -1:
        infer_kwargs['cropped_size'] = get_crop_size(game)
    log_dir = '{}/{}_{}_log'.format(EXPERIMENT_DIR, exp_name, n)
    data_path = os.path.join(log_dir, '{}_eval_data.pkl'.format(N_BINS))
    data_path_levels = os.path.join(log_dir, '{}_eval_data_levels.pkl'.format(N_BINS))
    if VIS_ONLY:
        if RENDER_LEVELS:
            eval_data_levels = pickle.load(open(data_path_levels, "rb"))
            eval_data_levels.render_levels()
            return
        eval_data = pickle.load(open(data_path, "rb"))
        # FIXME: just for backward compatibility
        eval_data.log_dir = log_dir
        eval_data.visualize_data(log_dir)
        eval_data.hamming_heatmap(None, eval_data.div_scores)
        return
    # no log dir, 1 parallel environment
    n_cpu = infer_kwargs.get('n_cpu')
    env, dummy_action_space, n_tools = make_vec_envs(env_name, representation, None, **infer_kwargs)
    model = load_model(log_dir, load_best=infer_kwargs.get('load_best'), n_tools=n_tools)
#   model.set_env(env)
    env.action_space = dummy_action_space
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
        control_bounds = env.envs[0].get_control_bounds()
    elif n_cpu > 1:
        env.remotes[0].send(('env_method', ('get_control_bounds', [], {})))  # supply args and kwargs
        control_bounds = env.remotes[0].recv()
    if not eval_controls:
        eval_controls = control_bounds.keys()
    ctrl_bounds = [(k, control_bounds[k]) for k in eval_controls]
    if len(ctrl_bounds) == 0 and DIVERSITY_EVAL:
        N_MAPS = 100
        N_TRIALS = 1
    # Hackish get initial states
    init_states = []
    for i in range(N_MAPS):
        env.envs[0].reset()
        # TODO: set initial states in either of these domains?
        if not (RCT or SC):
            init_states.append(env.envs[0].unwrapped._rep._map)
    N_EVALS = N_TRIALS * N_MAPS
    if len(ctrl_bounds) == 1:
        ctrl_name = ctrl_bounds[0][0]
        bounds = ctrl_bounds[0][1] 
        step_size = max((bounds[1] - bounds[0]) / (N_BINS[0] - 1), 1)
        eval_trgs = np.arange(bounds[0], bounds[1] + 1, step_size)
        level_images = []
        cell_scores = np.zeros((len(eval_trgs), 1))
        cell_ctrl_scores = np.zeros(shape=(len(eval_trgs), 1))
        cell_static_scores = np.zeros(shape=(len(eval_trgs), 1))
        level_tokens = []
        div_scores = np.zeros((len(eval_trgs), 1))
        for i, trg in enumerate(eval_trgs):
            trg_dict = {ctrl_name: trg}
            print('evaluating control targets: {}'.format(trg_dict))
            env.envs[0].set_trgs(trg_dict)
#           set_ctrl_trgs(env, {ctrl_name: trg})
            net_score, ctrl_score, static_score, level_image, tokens = eval_episodes(model, env, N_EVALS, n_cpu, init_states, log_dir, trg_dict, max_steps)
            div_score = div_calc(tokens)
            div_scores[i, 0] = div_score
            level_images.append(level_image)
            cell_scores[i] = net_score
            cell_ctrl_scores[i] = ctrl_score
            cell_static_scores[i] = static_score
            level_tokens.append(tokens)
        ctrl_names = (ctrl_name, None)
        ctrl_ranges = (eval_trgs, None)
        if RENDER_LEVELS:
            ims = np.hstack(level_images)
            image = Image.fromarray(ims)
            image.save(os.path.join(log_dir, levels_im_name.format(ctrl_names, N_BINS)))

    elif len(ctrl_bounds) >=2:
        ctrl_0, ctrl_1 = ctrl_bounds[0][0], ctrl_bounds[1][0]
        b0, b1 = ctrl_bounds[0][1], ctrl_bounds[1][1]
        step_0 = max((b0[1] - b0[0]) / (N_BINS[0] - 1), 1)
        step_1 = max((b1[1] - b1[0]) / (N_BINS[-1] - 1), 1)
        trgs_0 = np.arange(b0[0], b0[1]+0.5, step_0)
        trgs_1 = np.arange(b1[0], b1[1]+0.5, step_1)
        cell_scores = np.zeros(shape=(len(trgs_0), len(trgs_1)))
        div_scores = np.zeros(shape=(len(trgs_0), len(trgs_1)))
        cell_ctrl_scores = np.zeros(shape=(len(trgs_0), len(trgs_1)))
        cell_static_scores = np.zeros(shape=(len(trgs_0), len(trgs_1)))
        level_tokens = [[None]*len(trgs_0)]* len(trgs_1)  # Wait what?
        trg_dict = env.envs[0].static_trgs
        trg_dict = dict([(k, min(v)) if isinstance(v, tuple) else (k, v) for (k, v) in trg_dict.items()])
        level_images = []
        for i, t0 in enumerate(trgs_0):
            level_images_y = []
            for j, t1 in enumerate(trgs_1):
                ctrl_trg_dict = {ctrl_0: t0, ctrl_1: t1}
                trg_dict.update(ctrl_trg_dict)
                print('evaluating control targets: {}'.format(trg_dict))
    #           set_ctrl_trgs(env, {ctrl_name: trg})
                net_score, ctrl_score, static_score, level_image, tokens = eval_episodes(model, env, N_EVALS, n_cpu, init_states, log_dir, trg_dict, max_steps)
                level_images_y.append(level_image)
                cell_scores[i, j] = net_score
                cell_ctrl_scores[i, j] = ctrl_score
                cell_static_scores[i, j] = static_score
                div_score = div_calc(tokens)
                div_scores[i, j] = div_score
#               level_tokens[j][i] = tokens
            if RENDER_LEVELS:
                level_images.append(np.hstack(level_images_y))
#           level_tokens.append(tokens)
        ctrl_names = (ctrl_0, ctrl_1)
        ctrl_ranges = (trgs_0, trgs_1)
        image = None
        if RENDER_LEVELS:
            image = np.vstack(level_images[::-1])
            image = Image.fromarray(image)
            image.save(os.path.join(log_dir, levels_im_name.format(ctrl_names, N_BINS)))


    elif len(ctrl_bounds) == 0:
        N_BINS = None
        levels_im_name = "{}_{}-bins_levels.png"
        level_images = []
        cell_scores = np.zeros(shape=(1,1))
        div_scores = np.zeros(shape=(1,1))
        cell_static_scores = np.zeros(shape=(1,1))        
        cell_ctrl_scores = np.zeros(shape=(1,1))
        level_tokens = None 
        ctrl_names = None

        if DIVERSITY_EVAL:
            n_row = 1
            n_col = 1
        else:
            n_row = 2
            n_col = 5

        for i in range(n_row):
            level_images_y = []
            for j in range(n_col):
                net_score, ctrl_score, static_score, level_image, tokens = eval_episodes(model, env, N_EVALS, n_cpu, init_states, log_dir, env.envs[0].unwrapped._prob.static_trgs, max_steps)
                level_images_y.append(level_image)
                cell_scores[0,0] = net_score
                div_score = np.sum([np.sum(a != b) for a in tokens for b in tokens]) / (len(tokens) * (len(tokens) - 1))
                div_score = div_score / (map_width * map_width)
                div_scores[0,0] = div_score
            if RENDER_LEVELS:
                level_images.append(np.hstack(level_images_y))
        if RENDER_LEVELS:
            image = np.vstack(level_images[::-1])
            image = Image.fromarray(image)
            image.save(os.path.join(log_dir, levels_im_name.format(ctrl_names, N_BINS)))

        ctrl_ranges = None
    
    levels_im_path = os.path.join(log_dir, levels_im_name.format(ctrl_names, N_BINS))
    if not RENDER_LEVELS:
        image = None
    eval_data_levels = EvalData(ctrl_names, ctrl_ranges, cell_scores, cell_ctrl_scores, cell_static_scores, log_dir, levels_image=image, levels_im_path=levels_im_path)
    pickle.dump(eval_data_levels, open(data_path_levels, 'wb'))
    if not RENDER_LEVELS:
        eval_data = EvalData(ctrl_names, ctrl_ranges, cell_scores, cell_ctrl_scores, cell_static_scores, div_scores, log_dir)
        pickle.dump(eval_data, open(data_path, "wb"))
        eval_data.visualize_data(log_dir)

    else:
        levels_im_path = os.path.join(log_dir, levels_im_name.format(ctrl_names, N_BINS))
        eval_data_levels = EvalData(ctrl_names, ctrl_ranges, cell_scores, cell_ctrl_scores, cell_static_scores, div_scores, log_dir, levels_image=image, levels_im_path=levels_im_path)
        pickle.dump(eval_data_levels, open(data_path_levels, 'wb'))
        eval_data_levels.render_levels()

    if DIVERSITY_EVAL:
        eval_data = eval_data_levels
        if len(ctrl_bounds) == 0:
            eval_data.save_stats(div_scores)
        else:
            eval_data.hamming_heatmap(level_tokens, div_scores=div_scores)


def eval_episodes(model, env, n_trials, n_envs, init_states, log_dir, trg_dict, max_steps):
    env.envs[0].set_trgs(trg_dict)
    eval_scores = np.zeros(n_trials)
    eval_ctrl_scores = np.zeros(n_trials)
    eval_static_scores = np.zeros(n_trials)
    n = 0
    # FIXME: why do we need this?
    tokens = []
    while n < n_trials:
        if not (RCT or SC):
            env.envs[0].set_map(init_states[n % N_MAPS])
        elif SC:
            # Resize gui window for simcity
            env.envs[0].win1.editMapView.changeScale(0.77)
            env.envs[0].win1.editMapView.centerOnTile(20, 16)
        obs = env.reset()
#       epi_rewards = np.zeros((max_step, n_envs))
        i = 0
        # note that this is weighted loss
        init_loss = env.envs[0].get_loss()
        init_ctrl_loss = env.envs[0].get_ctrl_loss()
        init_static_loss = env.envs[0].get_static_loss()
        done = False
        while not done:
#           if i == max_steps - 1:
            if True:
                if RENDER_LEVELS:
                    image = env.render('rgb_array')
                    if SC and i == max_steps - 1:
                        #FIXME lmao fucking stupid as all heck
                        im_path = os.path.join(log_dir, '{}_level.png'.format(trg_dict))
                        image = env.envs[0].win1.editMapView.buffer.write_to_png(im_path)
                        em = env.envs[0].win1.editMapView
                        image = Image.open(im_path)
                        image = np.array(image)
                        print(image.shape)
                        image = image[:, 400:-400, :]
                        print(image.shape)
                    if RCT and i == max_steps - 1:
                        image = Image.fromarray(image.transpose(1, 0, 2))
                final_loss = env.envs[0].get_loss()
                final_ctrl_loss = env.envs[0].get_ctrl_loss()
                final_static_loss = env.envs[0].get_static_loss()
                if not (SC or RCT):
                    curr_tokens = env.envs[0].unwrapped._rep._map
                elif SC:
                    curr_tokens = env.envs[0].state.argmax(axis=0)
                elif RCT:
                    curr_tokens = env.envs[0].rct_env.park.map[0]
            action, _ = model.predict(obs)
            obs, rewards, done, info = env.step(action)
#           epi_rewards[i] = rewards
            i += 1
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
    print('eval score: {}'.format(eval_score))
    print('control score: {}'.format(eval_ctrl_score))
    print('static score: {}'.format(eval_static_score))
    if RENDER_LEVELS and not SC:
        # we hackishly save it for SC up above already
        image.save(os.path.join(log_dir, '{}_level.png'.format(trg_dict)))
        level_image = asarray(image)
    else:
        level_image = None
    return eval_score, eval_ctrl_score, eval_static_score, level_image, tokens






class EvalData():
    def __init__(self, ctrl_names, ctrl_ranges, cell_scores, cell_ctrl_scores, cell_static_scores, div_scores, log_dir, levels_image=None, levels_im_path=None):
        self.ctrl_names = ctrl_names
        self.ctrl_ranges = ctrl_ranges
        self.cell_scores = cell_scores
        self.cell_ctrl_scores = cell_ctrl_scores
        self.cell_static_scores = cell_static_scores
        self.div_scores = div_scores
        self.levels_image = levels_image
        self.levels_im_path = levels_im_path
        self.log_dir = log_dir
        
    def visualize_data(self, log_dir):

        def create_heatmap(title, data):
            fig, ax = plt.subplots()
            # percentages from ratios
            data = data * 100
            data = np.clip(data, -100, 100)
#           data = data.T
            if data.shape[1] == 1:
                data = data.T
                fig.set_size_inches(10, 2)
                tick_idxs = np.arange(0, cell_scores.shape[0], max(1, (cell_scores.shape[0] // 10)))
                ticks = np.arange(cell_scores.shape[0])
                ticks = ticks[tick_idxs]
                ax.set_xticks(ticks)
                labels = np.array([int(round(x, 0)) for (i, x) in enumerate(ctrl_ranges[0])])
                labels = labels[tick_idxs]
                ax.set_xticklabels(labels)
                if BAR_CHART:
#                   low = data[0].min()
                    low = 0
                    high = 100
                    plt.ylim([low, high])
                    ax.bar(ticks, data[0])
                    plt.xlabel(ctrl_names[0])
                else:
                    ax.set_yticks([])
            else:
                data = data[::-1,:]
                ax.set_xticks(np.arange(cell_scores.shape[1]))
                ax.set_yticks(np.arange(cell_scores.shape[0]))
                ax.set_xticklabels([int(round(x, 0)) for x in ctrl_ranges[1]])
                ax.set_yticklabels([int(round(x, 0)) for x in ctrl_ranges[0][::-1]])
            # Create the heatmap
#           im = ax.imshow(data, aspect='auto', vmin=-100, vmax=100)
            if not BAR_CHART or data.shape[0] != 1:
                im = ax.imshow(data, aspect='auto', vmin=0, vmax=100)

                #Create colorbar
                cmap = colors.ListedColormap(['b','r','y','r'])
                cbar = ax.figure.colorbar(im, ax=ax)
                cbar.ax.set_ylabel("", rotation=90, va="bottom")

                # We want to show all ticks...
                # ... and label them with the respective list entries
                plt.xlabel(ctrl_names[1])
                plt.ylabel(ctrl_names[0])

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            ax.set_title(title)
            fig.tight_layout()

            plt.savefig(os.path.join(log_dir, "{}_{}.png".format(ctrl_names, title.replace("%", ""))))
#           plt.show()

        ctrl_names = self.ctrl_names
        ctrl_ranges = self.ctrl_ranges
        cell_scores = self.cell_scores
        cell_ctrl_scores = self.cell_ctrl_scores
        cell_static_scores = self.cell_static_scores

        title = "All goals (mean progress, %)"
        create_heatmap(title, cell_scores)

        title = "Controlled goals (mean progress, %)"
        create_heatmap(title, cell_ctrl_scores)

        title = "Fixed goals (mean progress, %)"
        create_heatmap(title, cell_static_scores)

        title = "Hamming distances"
        self.save_stats()

    def pairwise_hamming(self, a, b):
        return np.sum(a != b)

    def hamming_heatmap(self, level_tokens, div_scores=None):
        if N_MAPS==1:
            return
        
        fig, ax = plt.subplots()
        title = "Diversity"

        if div_scores is not None:
            hamming_scores = div_scores.T
        else:
            #get the hamming distance between all possible pairs of chromosomes in each cell
            #1) make the evaldata function have the tilemap for each env in each bucket.
            #2) feed THAT info to this function.
            print(len(level_tokens))
            print(len(level_tokens[0]))
            print(len(level_tokens[0][0]))
            if type(level_tokens[0][0])==list:
                hamming_scores = np.zeros(shape=(len(level_tokens[0]), len(level_tokens)))
                for i, row in enumerate(level_tokens):
                    for j, col in enumerate(level_tokens[i]):
                        hamming = 0
                        counter = 0
                        for k in range(len(col)-1):
                            for l in range(k+1,len(col)):
                                print("index: ", k, l)
                                hamming += self.pairwise_hamming(col[k], col[l])
                                counter+=1
                        # Division by zero can happen here, why?
                        if counter > 0:
                            hamming = hamming/counter
                        print(hamming_scores.shape)
                        hamming_scores[j,i] = hamming
            else:
                hamming_scores = np.zeros(shape=(len(level_tokens), 1))
                for i in range(len(level_tokens[0])):
                    for j in range(len(level_tokens)):
                        print("bin ", j)
                        print(level_tokens[j][i][0])

                for i, tokens in enumerate(level_tokens):
                    hamming = 0
                    counter = 0
                    for j in range(len(tokens)-1):
                        for k in range(j+1, len(tokens)):
                            print("index: ", j, k)
                            hamming += self.pairwise_hamming(tokens[j], tokens[k])
                            counter+=1
                    hamming = hamming/counter
                    hamming_scores[i] = hamming
            hamming_scores = hamming_scores.T
            print("ctr", counter)
        print(hamming_scores)

        if hamming_scores.shape[0]==1:
            fig.set_size_inches(10, 2)
            tick_idxs = np.arange(0, self.cell_scores.shape[0], max(1, (self.cell_scores.shape[0] // 10)))
            ticks = np.arange(self.cell_scores.shape[0])
            ticks = ticks[tick_idxs]
            ax.set_xticks(ticks)
            labels = np.array([int(round(x, 0)) for (i, x) in enumerate(self.ctrl_ranges[0])])
            labels = labels[tick_idxs]
            ax.set_xticklabels(labels)
            if BAR_CHART:
                low = 0
                high = 0.7
#               low = hamming_scores[0].min()
#               high = hamming_scores[0].max()
                plt.ylim([low, high])
                plt.bar(ticks, hamming_scores[0], color='purple')
                plt.xlabel(self.ctrl_names[0])
            else:
                ax.set_yticks([])
        else:
            hamming_scores = hamming_scores[::-1,:]
            ax.set_xticks(np.arange(self.cell_scores.shape[1]))
            ax.set_yticks(np.arange(self.cell_scores.shape[0]))
            ax.set_xticklabels([int(round(x, 0)) for x in self.ctrl_ranges[1]])
            ax.set_yticklabels([int(round(x, 0)) for x in self.ctrl_ranges[0][::-1]])
        if not BAR_CHART or hamming_scores.shape[0] != 1:
            # Create the heatmap

            #Create colorbar
#           cmap = colors.ListedColormap(['r','y','b','r'])
            cmap = plt.get_cmap('inferno')
#           im = ax.imshow(hamming_scores, aspect='auto', cmap=cmap, vmin=0, vmax=0.7)
            im = ax.imshow(hamming_scores, aspect='auto', cmap=cmap)
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("", rotation=90, va="bottom")


            # We want to show all ticks...
            # ... and label them with the respective list entries
            plt.xlabel(self.ctrl_names[1])
            plt.ylabel(self.ctrl_names[0])

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        ax.set_title(title)
        fig.tight_layout()

        plt.savefig(os.path.join(self.log_dir, "{}_{}.png".format(self.ctrl_names, title.replace("%", ""))))


    def save_stats(self, div_scores=np.zeros(shape=(1,1))):
        scores = {
            'net_score': self.cell_scores.mean(),
            'ctrl_score': self.cell_ctrl_scores.mean(),
            'fixed_score': self.cell_static_scores.mean(),
            'diversity_score': div_scores.mean(),
        }
        with open(os.path.join(self.log_dir, 'scores.json'), 'w') as fp:
            json.dump(scores, fp)

    def render_levels(self):
        ctrl_names = self.ctrl_names
        fig, ax = plt.subplots()
        ax.imshow(self.levels_image)
    #       ax.axis["xzero"].set_axisline_style("-|>")
       #plt.tick_params(
       #    axis='x',
       #    which='both',
       #    bottom=False,
       #    top=False,
       #    labelbottom=False)

        if ctrl_names[1] is None:
            plt.xlabel(ctrl_names[0])
            # wut
            im_width = np.array(self.levels_image).shape[1] / self.cell_scores.shape[0]
            plt.xticks(np.arange(self.cell_scores.shape[0]) * im_width + im_width / 2, labels=[int(round(x, 0)) for x in self.ctrl_ranges[0]])
            pad_inches = 0
            hspace = 0
            bottom = 0
            plt.tick_params(
                axis='y',
                which='both',
                left=False,
                right=False,
                labelleft=False)
        else:
            plt.xlabel(ctrl_names[0])
            plt.ylabel(ctrl_names[1])
            im_width = np.array(self.levels_image).shape[1] / self.cell_scores.shape[1]
            im_height = np.array(self.levels_image).shape[0] / self.cell_scores.shape[0]
            plt.xticks(np.arange(self.cell_scores.shape[0]) * im_width + im_width / 2, labels=[int(round(x, 0)) for x in self.ctrl_ranges[0]])
            plt.yticks(np.arange(self.cell_scores.shape[1]) * im_height + im_height / 2, labels=[int(round(y, 0)) for y in self.ctrl_ranges[1][::-1]])
#           ax.set_xticklabels([round(x, 1) for x in ctrl_ranges[0]])
#           ax.set_yticklabels([round(x, 1) for x in ctrl_ranges[1][::-1]])

            pad_inches = 0
            hspace = 0
            bottom = 0.05
       #plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = bottom, right = 1, left = 0, 
                    hspace = hspace, wspace = 0)
        plt.margins(0,0)
       #plt.gca().xaxis.set_major_locator(plt.NullLocator())
       #plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(self.levels_im_path, bbox_inches = 'tight',
            pad_inches = pad_inches)
        plt.show()




#NOTE: let's not try multiproc how about that :~)

#def eval_episodes(model, env, n_trials, n_envs):
#    eval_scores = np.zeros(n_trials)
#    n = 0
#    # FIXME: why do we need this?
#    env.reset()
#    while n < n_trials:
#
#        obs = env.reset()
##       epi_rewards = np.zeros((max_step, n_envs))
#        i = 0
##       env.remotes[0].send(('env_method', ('get_metric_vals', [], {})))  # supply args and kwargs
##       init_metric_vals = env.remotes[0].recv()
#        [remote.send(('env_method', ('get_loss', [], {}))) for remote in env.remotes]
#        # note that this is weighted loss
#        init_loss = np.sum([remote.recv() for remote in env.remotes])
#        dones = np.array([False])
#        while not dones.all():
#            action, _ = model.predict(obs)
#            obs, rewards, dones, info = env.step(action)
##           epi_rewards[i] = rewards
#            i += 1
#        # since reward is weighted loss
#        final_loss = np.sum(rewards)
#        # what percentage of loss (distance from target) was recovered?
#        score = (final_loss - init_loss) / abs(init_loss)
##       env.remotes[0].send(('env_method', ('get_metric_vals', [], {})))  # supply args and kwargs
##       final_metric_vals = env.remotes[0].recv()
#        eval_scores[n] = score
#        n += n_envs
#    return eval_scores.mean()
#
#def set_ctrl_trgs(env, trg_dict):
#    [remote.send(('env_method', ('set_trgs', [trg_dict], {}))) for remote in env.remotes]

from arguments import get_args
args = get_args()
args.add_argument('--vis_only',
        help='Just load data from previous evaluation and visualize it.',
        action='store_true',
        )
args.add_argument('--eval_controls',
        help='Which controls to evaluate and visualize.',
        nargs='+',
        default=[],
        )
args.add_argument('--n_maps',
        help='Number maps on which to simulate in each cell.',
        default=10,
        type=int,
        )
args.add_argument('--n_trials',
        help='Number trials for which to simulate on each map.',
        default=1,
        type=int,
        )
#args.add_argument('--step_size',
#        help='Bin size along either dimension.',
#        default=20,
#        type=int,
#        )
args.add_argument('--n_bins',
        help='How many bins along each dimension (a sequence of ints).',
        nargs='+',
        type=int,
        default=(10,10),
        )
args.add_argument('--render_levels',
        help='Save final maps (default to only one eval per cell)',
        action='store_true',
        )
#args.add_argument('--diversity_eval',
#        help='Evaluate average pairwise hamming distance per cell',
#        action='store_true',
#        )
from arguments import parse_pcgrl_args
opts = parse_pcgrl_args(args)
global VIS_ONLY 
VIS_ONLY = opts.vis_only
#DIVERSITY_EVAL = opts.diversity_eval

# For locating trained model
global EXPERIMENT_ID
global EXPERIMENT_DIR
#EXPERIMENT_DIR = 'hpc_runs/runs'
if not opts.HPC:
    EXPERIMENT_DIR = 'rl_runs'
else:
    EXPERIMENT_DIR = 'hpc_runs'
EXPERIMENT_ID = opts.experiment_id
problem = opts.problem
if 'RCT' in problem:
    RCT = True
else:
    RCT = False
if 'Micropolis' in problem:
    SC = True
else:
    SC = False
representation = opts.representation
conditional = opts.conditional
midep_trgs = opts.midep_trgs
ca_action = opts.ca_action
alp_gmm = opts.alp_gmm
kwargs = {
       #'change_percentage': 1,
       #'target_path': 105,
       #'n': 4, # rank of saved experiment (by default, n is max possible)
        }

RENDER_LEVELS = opts.render_levels
change_percentage = opts.change_percentage
#NOTE: For now rendering levels and doing any sort of evaluation are separate processes because we don't need to render all that and it would be inefficient but we do need many runs for statistical significance. Pray for representative levels.
DIVERSITY_EVAL = not RENDER_LEVELS
if problem == 'sokobangoal':
    map_width = 5
else:
    map_width = 16

if conditional:
    max_step = opts.max_step
#   if max_step is None:
#       if RENDER_LEVELS:
#           max_step = 10000
#       else:
#           max_step = 5000
    cond_metrics = opts.conditionals

    if ca_action:
        max_step = 50
    change_percentage = 1.0
else:
    max_step = None
    cond_metrics = None
    change_percentage

# For inference
infer_kwargs = {
        'change_percentage': change_percentage,
       #'target_path': 200,
        'conditional': cond_metrics,
        'cond_metrics': cond_metrics,
        'max_step': max_step,
        'render': opts.render,
        # TODO: multiprocessing
#       'n_cpu': opts.n_cpu,
        'n_cpu': 1,
        'load_best': opts.load_best,
        'midep_trgs': midep_trgs,
        'infer': True,
        'ca_action': ca_action,
        'map_width': map_width,
        'eval_controls': opts.eval_controls,
        'cropped_size': opts.crop_size,
        }

global N_BINS
N_BINS = tuple(opts.n_bins)

if RENDER_LEVELS and not DIVERSITY_EVAL:
    #FIXME: this is not ob
    N_MAPS = 1
    N_TRIALS = 1
else:
    N_MAPS = opts.n_maps
#   N_TRIALS = opts.n_trials
    N_TRIALS = 1

if __name__ == '__main__':

    evaluate(problem, representation, infer_kwargs, **kwargs)
#   evaluate(test_params, game, representation, experiment, infer_kwargs, **kwargs)
#   analyze()