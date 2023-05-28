import argparse
import json
import operator
import os
from pathlib import Path
import re
import csv
from pdb import set_trace as TT
from typing import Dict, List
import hydra

import numpy as np
from matplotlib import pyplot as plt
from omegaconf import DictConfig, ListConfig
import pandas as pd
from control_pcgrl.configs.config import Config, CrossEvalConfig

from control_pcgrl.rl.utils import get_log_dir, PROB_CONTROLS, validate_config
from tex_formatting import newline, pandas_to_latex

# OVERLEAF_DIR = "/home/sme/Dropbox/Apps/Overleaf/Evolving Diverse NCA Level Generators -- AIIDE '21/tables"

# Map names of metrics recorded to the names we want to display in a table

RUNS_DIR = os.path.join(Path(__file__).parent.parent.parent, 'rl_runs')
EVAL_DIR = os.path.join(Path(__file__).parent.parent.parent, 'rl_eval')

keys = [
    "task", 
    "representation", 
    "model",
    "n_aux_tiles",
    "max_board_scans",
    "controls",
    "lr",
    "exp_id",
    # "controls", 
    # "alp_gmm", 
    # "change_percentage"
]

local_controls = {
    "binary_ctrl": [
        ["regions", "path-length"],
        ["regions"],
        ["path-length"],
        # ['emptiness', 'path-length'],
        # ["symmetry", "path-length"]
    ],
    "zelda_ctrl": [
        ["nearest-enemy", "path-length"],
        ["nearest-enemy"],
        ["path-length"],
        # ["emptiness", "path-length"],
        # ["symmetry", "path-length"],
    ],
    "sokoban_ctrl": [
        # ["crate"],
        ["crate", "sol-length"],
        ["sol-length"],
        # ["emptiness", "sol-length"],
        # ["symmetry", "sol-length"],
    ],
    "smb_ctrl": [
        # ['enemies', 'jumps'],
        # ["emptiness", "jumps"],
        # ["symmetry", "jumps"],
    ],
    "RCT": [
        # ['income'],
    ],
}

header_text = {
    "zelda_ctrl": "zelda",
    "binary_ctrl": "binary",
    "sokoban_ctrl": "sokoban",
    "NONE": "---",
    "change_percentage": newline("change", "percentage"), 
#   'net_score (mean)': newline('target', 'progress'),
    'net_score (mean)': 'pct. targets reached ',
#   '(controls) net_score (mean)': newline('\\textit{control}', 'net score'),
    "diversity_score (mean)": "diversity",
#   "(controls) diversity_score (mean)": newline('\\textit{control}', 'diversity'),
#   'ctrl_score (mean)': newline('control', 'success'),
    'ctrl_score (mean)': 'pct. targets reached',
#   '(controls) fixed_score (mean)': newline('\\textit{control}', 'static score'),
#   "alp_gmm": newline("ALP", "GMM"),
    "alp_gmm": newline("control", "regime"),
    "controls": newline("learned", " controls"),
}

# flatten the dictionary here


def bold_extreme_values(data, data_max=-1):
    data_max = data_max
    data = int(data)
#   print(data)
    if data == data_max:
#       return "\\bfseries {:.2f}".format(data)
        return "\\bfseries {}".format(data)

    else:
#       return "{:.1f}".format(data)
        return "{}".format(data)

#   return data
#   return "{}".format(data)


def flatten_stats(stats, controllable=False):
    flat_stats = {}

    # def add_key_val(key, val):
#         if controllable and key != "% train archive full":
# #           key = "(controls) " + key
#             if key in header_text:
#                 key = header_text[key]
#             key = ', '.join([c for c in stats['controls'] if c is not None]) + '||' + key

        # if "%" in key:
            # val *= 100
        # elif "playability" in key:
            # val /= 10

        # if key in header_text:
            # key = header_text[key]
        # flat_stats[key] = val

    for k, v in stats.items():
        if isinstance(v, dict):
            key_0 = k

            for k1, v1 in v.items():
                key = "{} ({})".format(key_0, k1)
                value = v1
                flat_stats[key] = value
        else:
            flat_stats[k] = v

    return flat_stats


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def cross_evaluate(cross_eval_config: Config, sweep_configs: List[Config], sweep_params: Dict[str, str]):
    """Collect results generated when evaluating trained models under different conditions.
    Args:
        cross_eval_config (CrossEvalConfig): The cross-evaluation config
        sweep_configs (List[EvalConfig]): EvalConfigs corresponding to evaluations
        sweep_params (Dict[str, str]): The eval/train hyperparameters being swept over in the cross-evaluation
    """
    # validate_config(cross_eval_config)
    # [validate_config(c) for c in sweep_configs]

    experiment_0 = sweep_configs[0]

    col_headers = ['episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 'episode_len_mean', 
                   'episodes_this_iter', 'total_steps']
    # TODO: Automate these!
    col_headers += ['path-length_max', 'path-length_mean', 'path-length_min']
    row_headers = [k for k in sweep_params.keys()]

    # row_headers_sorted = ['model', 'act_window', 'exp_id']
    # sort_map = {k: i for i, k in enumerate(row_headers_sorted)}

    # row_headers = sorted(row_headers, key=sort_map.__getitem__)

    rows = []
    vals = []

    for experiment in sweep_configs:
        exp_path = experiment.log_dir
        path = os.path.join(exp_path, "eval_stats.json")

        if not os.path.isfile(path):
            print(f"No eval_stats.json found in {path}")
            continue
            
        print(f"Loading eval stats from {path}")

        with open(path, "r") as f:
            stats = json.load(f)
            # stats = flatten_stats(stats)
            stats = flatten_dict(stats)
     
        row = []

        for k in row_headers:

            v = operator.attrgetter(k)(experiment)

            # If a subconfig, take its name(?? HACK)
            if isinstance(v, DictConfig):
                v_name = v.name
                if v_name is None:
                    if k == 'model':
                        v_name = 'Default'
                    row.append(v_name)
                else:
                    row.append(v.name.replace('_', ' '))

            elif isinstance(v, ListConfig):
                # Important that we turn this into a string lest terrible weirdness ensue.
                row.append(str(tuple([vi for vi in v])))

            else:
                row.append(str(v))

        rows.append(row)

        exp_stats = [stats[k] for k in col_headers if k in stats]
        vals.append(exp_stats)

    # iterate col_headers and replace "_" with " " to avoid latex errors
    col_headers = [h.replace("_", " ") for h in col_headers]
    col_headers = [h.replace(".", " ") for h in col_headers]
    row_headers = [h.replace("_", " ") for h in row_headers]
    row_headers = [h.replace(".", " ") for h in row_headers]

    # Make a multi-indexed dataframe
    rows = pd.MultiIndex.from_tuples(rows, names=row_headers)

    df = pd.DataFrame(vals, index=rows, columns=col_headers)

    # # Replace every `_` with a `\_` to avoid latex errors, in rows...
    # df.index = df.index.map(lambda x: [str(i).replace("_", " ") for i in x])
    # df.index = df.index.map(lambda x: [str(i).replace("[", "") for i in x])
    # # and in columns.
    # df.columns = df.columns.map(lambda x: x.replace("_", " "))
    # # And in the values, format to 2 decimal places
    # df = df.applymap(lambda x: "{:.2f}".format(x))

    def write_data(df, tex_name):

        # Save the dataframe
        df.to_csv(os.path.join(EVAL_DIR, f"{tex_name}.csv"))

        # Save the df as latex
        # df = df.applymap(bold_extreme_values)
        # df = df.applymap(lambda x: "{:.2f}".format(x))

        # tex_name = cross_eval_config.name

        # Save the df as latex
        # df.to_latex(
        #     os.path.join(EVAL_DIR, "cross_eval.tex"),
        #     column_format="l" + "c" * len(col_headers),
        #     multirow=True,
        #     escape=False,
        # )
        pandas_to_latex(
            df, 
            os.path.join(EVAL_DIR, tex_name + '.tex'),
            # multirow=True, 
            index=True, 
            header=True,
            vertical_bars=True,
            # columns=col_indices, 
            multicolumn=True, 
            # multicolumn_format='c|',
            right_align_first_column=False,
            # bold_rows=True,
        )

        tables_tex_fname = os.path.join(EVAL_DIR, "tables.tex")

        # Replace the line in `tables.tex` that says `\input{.*}` with `\input{tables_tex_fname.tex}` with regex
        re.sub(r"\\input{.*}", "poo", os.path.join(EVAL_DIR, "tables.tex"))

        os.system(f"pdflatex {tables_tex_fname}")

        # Move the output pdf int rl_eval
        # os.system(f"mv {tables_tex_fname.replace('.tex', '.pdf')} {EVAL_DIR}")

        return

    write_data(df, 'cross_eval')

    row_headers.remove('exp id')

    # Average over exp_id (no cooperating currently)
    # df = df.groupby(row_headers).mean(numeric_only=True).reset_index()
    # TODO: standard deviation

    # Average/stds the ugly manual way
    new_rows = []
    new_row_names = []
    for i in range(df.shape[0]):
        row = df.iloc[i]
        name = row.name
        if name[:-1] in new_row_names:
            continue
        new_row_names.append(name[:-1])

        # For some reason this is necessary even though we've already done this when processing row headers above?
        # new_name = []
        # for ni in name:
        #     if isinstance(ni, ListConfig):
        #         ni = tuple([nii for nii in ni])
        #     new_name.append(ni)
        # name = new_name

        repeat_exps = df.loc[name[:-1]]
        # eval_qd_scores.append(repeat_exps[('Evaluation', 'QD score')].to_numpy())
        mean_exp = repeat_exps.mean(axis=0)
        std_exp = repeat_exps.std(axis=0)
        mean_exp = [(i, e) for i, e in zip(mean_exp, std_exp)]
        new_rows.append(mean_exp)

    index = pd.MultiIndex.from_tuples(new_row_names, names=row_headers)
    columns = df.columns
    ndf = pd.DataFrame(new_rows, index=index, columns=col_headers)
    # ndf = ndf.append(new_rows)
    # new_col_indices = pd.MultiIndex.from_tuples(df.columns)
    # new_row_indices = pd.MultiIndex.from_tuples(new_row_names)
    # ndf.index = new_row_indices
    # ndf.columns = new_col_indices
    # ndf.index.names = df.index.names[:-1]
    ndf = ndf

    write_data(ndf, 'cross_eval_aggregate_raw')

    drop_cols = ['total steps', 'episode len mean', 'episodes this iter']
    ndf = ndf.drop(columns=drop_cols)

    write_data(ndf, 'cross_eval_aggregate')
    return

    # batch_exp_name = settings_list[0]["exp_id"]
    #   if batch_exp_name == "2":
    #   elif batch_exp_name == "1":
    #       EVO_DIR = "evo_runs_06-13"
    #       RL_DIR = "evo_runs_06-14"
    #   ignored_keys = set(
    #       (
    #           "exp_name",
    #           "evaluate",
    #           "show_vis",
    #           "visualize",
    #           "render_levels",
    #           "multi_thread",
    #           "play_level",
    #           "evaluate",
    #           "save_levels",
    #           "cascade_reward",
    #           "model",
    #           "n_generations",
    #           "render",
    #           "infer",
    #       )
    #   )
    #   keys = []

    #   for k in settings_list[0].keys():
    #       if k not in ignored_keys:
    #           keys.append(k)
    columns = None
    data = []
    vals = []

    for i, settings in enumerate(sweep_configs):
        val_lst = []

        controllable = False
        for k in keys:
            val = settings[k]
            if k == 'controls':
                if k != ['NONE', 'NONE']:
                    controllable = True
            if isinstance(settings[k], list):
                if len(settings[k]) < 2:
                    val = ", ".join(settings[k])
                else:
                    val = newline(settings[k][0]+', ', val[1])
            elif k == 'alp_gmm':
                if not controllable:
                    val = ''
                elif val:
                    val = 'ALP-GMM'
                else:
                    val = newline('uniform', 'random')
            if isinstance(val, str):
                val = val.replace('_', ' ')
            val_lst.append(val)
        # args = parse_args(load_args=settings)
        # arg_dict = vars(args)
        # dict to namespace
        args = argparse.Namespace(**settings)
        arg_dict = vars(args)
        # FIXME: well this is stupid
        exp_name = get_log_dir(args) + '_' + str(arg_dict["exp_id"]) + '_log'  # FIXME: this should be done elsewhere??
        arg_dict["cond_metrics"] = arg_dict.pop("controls")
        # NOTE: For now, we run this locally in a special directory, to which we have copied the results of eval on
        # relevant experiments.
        exp_name = os.path.join(RUNS_DIR, exp_name)
        # eval_dir = os.path.join(exp_name, "eval")
        eval_dir = exp_name
        eval_stats = json.load(open(os.path.join(eval_dir, "eval_stats.json"), 'r'))
        if not os.path.isdir(eval_dir):
            print("skipping evaluation of experiment due to missing directory: {}".format(eval_dir))
            continue
        ctrl_stats_files = [f for f in os.listdir(eval_dir) if re.match(r"scores_.*_ctrlTrgs.json", f)]
        # stats_f = os.path.join(eval_dir, "scores_ctrlTrgs.json")
        fixTrgs_stats_f = os.path.join(eval_dir, "scores_fixTrgs.json")

        if not ctrl_stats_files and os.path.isfile(fixTrgs_stats_f):
#           print(stats_f)
            print(
                "skipping evaluation of experiment due to missing stats file(s): {}".format(
                    exp_name
                )
            )

            continue
        ctrl_stats_files = [os.path.join(eval_dir, f) for f in ctrl_stats_files]
        # TODO: This is just here for convenience, but not really a part of cross-eval, semantically-speaking
        if not no_plot:
            plot_csv(exp_name)
        vals.append(tuple(val_lst))
        data.append([])
        flat_stats = flatten_stats(eval_stats)
        # fixLvl_stats = json.load(open(fixTrgs_stats_f, "r"))
        # flat_stats = flatten_stats(fixLvl_stats)
        # for stats_f in ctrl_stats_files:
        #     stats = json.load(open(stats_f, "r"))
        #     flat_stats.update(flatten_stats(stats, controllable=True))

        if columns is None:
            columns = list(flat_stats.keys())

        for j, c in enumerate(columns):
            if c not in flat_stats:
                data[-1].append("N/A")
            else:
                data[-1].append(flat_stats[c])

    tuples = vals
    # Rename headers
    new_keys = []
    # alp_gmm is always "False" for non-controllable agents. But we don't want to display them as having any control regime.
    # controls_id = keys.index('controls')
    # regime_id = keys.index('alp_gmm')
    # for i, tpl in enumerate(tuples):
    #     if tpl[controls_id] == 'NONE':
    #         tpl = list(tpl)
    #         tpl[regime_id] = '---'
    #         tpl = tuple(tpl)
    #         tuples[i] = tpl

    for k in keys:
        if k in header_text:
            k = header_text[k]
        k = k.replace("_", " ")
        new_keys.append(k)
    
    for (i, lst) in enumerate(tuples):
        new_lst = []
        for v in lst:
            if v in header_text:
                new_lst.append(header_text[v])
            else:
                new_lst.append(v)
        tuples[i] = new_lst

    index = pd.MultiIndex.from_tuples(tuples, names=new_keys)
    #   df = index.sort_values().to_frame(index=True)
    # Hierarchical columns!
    col_tuples = []
    # for col in columns:
    #     if '||' not in col:
    #         col_tuples.append(('fixed targets', '---', col))
    #     else:
    #         controls = col.split('||')[0]
    #         col_tuples.append(('controlled targets', controls, col.split('||')[-1]))
    # columns = pd.MultiIndex.from_tuples(col_tuples, names=['', newline('evaluated', 'controls'), ''])
    # columns = pd.MultiIndex.from_tuples(col_tuples, names=['', 'evaluated controls', ''])
    df = pd.DataFrame(data=data, index=index, columns=columns)
#   df = df.sort_values(by=new_keys, axis=0)
#   new_keys = [tuple(t) for t in tuples]
#   df = df.sort_values(by=new_keys, axis=0)

    csv_name = r"{}/cross_eval_{}.csv".format(EVAL_DIR, batch_exp_name)
    html_name = r"{}/cross_eval_{}.html".format(EVAL_DIR, batch_exp_name)
    df.to_csv(csv_name)
    df.to_html(html_name)
#   print(df)

    columns = [c.replace('_', ' ') for c in columns]
    df_tex = pd.DataFrame(data=data, index=index, columns=columns)
    #   tex_name = r"{}/zelda_empty-path_cell_{}.tex".format(OVERLEAF_DIR, batch_exp_name)

    # TODO: dust off latex table-generation for new domains.

    # for p in ["binary", "zelda", "sokoban", "minecraft_3D_maze"]:
#   for p in ["binary", "zelda"]:
#   for p in ["binary"]:
    tex_name = os.path.join(EVAL_DIR, "cross_eval.tex")
# #       print(p)
#         p_name = p + '_ctrl'
#         lcl_conds = ['---'] + ['-'.join(pi) if len(pi) < 2 else newline(pi[0] + ', ', pi[1]) for pi in local_controls[p_name]]
# #       print(lcl_conds)
# #       df_tex = df_tex.loc[lcl_conds]
# #       df_tex = df_tex.sort_values(by=['ALP GMM'])
#         z_cols_fixed = [
#             header_text["net_score (mean)"],
#             header_text["diversity_score (mean)"],
#         ]
#         z_cols_ctrl = [
# #           col_keys["net_score (mean)"],
#             header_text["ctrl_score (mean)"],
# #           col_keys["(controls) fixed_score (mean)"],
#             header_text["diversity_score (mean)"],
#         ]
#         n_col_heads = len(z_cols_fixed)
#         z_cols = list(zip(['fixed targets'] * n_col_heads, ['---'] * n_col_heads, z_cols_fixed))
#         for ctrl_set in PROB_CONTROLS[p + '_ctrl']:
#             if 'sol-length' in ctrl_set:
#                 ctrl_set[ctrl_set.index('sol-length')] = 'solution-length'
#             n_col_heads = len(z_cols_ctrl)
#             z_cols += list(zip(['controlled targets'] * n_col_heads, [', '.join(ctrl_set)] * n_col_heads, z_cols_ctrl))
#         df_tex = df_tex[z_cols]
#         #   df_tex = df.drop(columns=z_cols)
# #       df_tex['fixed targets'] = df_tex['fixed targets'][z_cols[0:2]]
# #       df_tex['controlled targets'] = df_tex['fixed targets'][z_cols[2:]]
#         df_tex = df_tex * 100
#         df_tex = df_tex.astype(float).round(0)
#         dual_conds = ['---', lcl_conds[1]]
#         for k in z_cols:
#             print(k)
#             if k in df_tex:
#                 print(k)
# #               df_tex.loc[dual_conds][k] = df_tex.loc[dual_conds][k].apply(
# #                   lambda data: bold_extreme_values(data, data_max=df_tex.loc[dual_conds][k].max())
# #               )
#                 print(df_tex[k].max())
#                 df_tex[k] = df_tex[k].apply(
#                     lambda data: bold_extreme_values(data, data_max=df_tex[k].max())
#                 )
# #       df_tex = df_tex.round(2)
# #       df_tex.reset_index(level=0, inplace=True)
# #       print(df_tex)

# #       with open(tex_name, "w") as tex_f:
# #           col_widths = "p{0.5cm}p{0.5cm}p{0.5cm}p{0.5cm}p{0.5cm}p{0.5cm}p{0.8cm}p{0.8cm}p{0.8cm}"
    pandas_to_latex(
        df_tex,
#           df_tex.to_latex(
        tex_name,
        vertical_bars=True,
        index=True,
        bold_rows=True,
        header=True,
        columns=columns,
        multirow=True,
        multicolumn=True,
        multicolumn_format='c|',
        # column_format= 'r|' * len(index) + 'c|' * len(df_tex.columns),
        escape=False,
        # caption=("Performance of controllable {} level-generating agents with learning-progress-informed (ALP-GMM) and uniform-random control regimes, and baseline (single-objective) agents, with various change percentage allowances. Agents are tested both on a baseline task with metric targets fixed at their default values, and control tasks, in which controllable metric targets are sampled over a grid.".format(p)),
        # label={"tbl:{}".format(p)},
    )

    tables_tex_fname = os.path.join(EVAL_DIR, "tables.tex")
    os.system(f"pdflatex {tables_tex_fname}")

    #   # Remove duplicate row indices for readability in the csv
    #   df.reset_index(inplace=True)
    #   for k in new_keys:
    #       df.loc[df[k].duplicated(), k] = ''
    #   csv_name = r"{}/cross_eval_{}.csv".format(OVERLEAF_DIR, batch_exp_name)

def listolists_to_arr(a):
    b = np.empty([len(a), len(max(a, key=lambda x: len(x)))])
    b[:] = np.NaN
    for i, j in enumerate(a):
        b[i][0:len(j)] = j
    return b

def plot_csv(exp_name):
    monitor_files = [os.path.join(exp_name, f) for f in os.listdir(exp_name) if "monitor.csv" in f]
    g_rew_id, g_len_id, g_time_id = None, None, None
    g_reward_weights, g_lens, g_times = [], [], []
    for f in monitor_files:
        rewards, lens, times = [], [], []
        with open(f, newline='') as mf:
            reader = csv.reader(mf, delimiter=',', quotechar='|')
            for i, row in enumerate(reader):
                if i == 0:
                    # This contains one-off info like time start and env id
                    continue
                if i == 1:
                    # Get header info. stable-baselines determines how this is written. Being extra safe here.
                    rew_id, len_id, time_id = row.index('r'), row.index('l'), row.index('t')
                    if g_rew_id is None:
                        g_rew_id, g_len_id, g_time_id = rew_id, len_id, time_id
                    else:
                        assert g_rew_id == rew_id and g_len_id == len_id and g_time_id == time_id
                    continue
                if len(row) != 3:
                    print('row {} of monitor file {} is not the correct length: {}. Skipping'.format(i, f, row))
                    continue
                try:
                    row = [float(r) for r in row]
                except ValueError:
                    print('row {} of monitor file {} has invalid values: {}. Skipping.'.format(i, f, row))
                    continue
                rewards.append(row[g_rew_id])
                lens.append(row[g_len_id])
                times.append(row[g_time_id])
        g_reward_weights.append(rewards)
        g_lens.append(lens)
        g_times.append(times)
    if not g_reward_weights:
        return
    rew_arr, len_arr, time_arr = listolists_to_arr(g_reward_weights), listolists_to_arr(g_lens), listolists_to_arr(g_times)
    mean_rews, mean_lens, mean_times = np.nanmean(rew_arr, axis=0), np.nanmean(len_arr, axis=0), np.nanmean(time_arr, axis=0)
    plt.figure()
    mean_rews = np.convolve(mean_rews, [1/7 for i in range(7)])
    with open(os.path.join(exp_name, 'train_time_stats.json'), 'w') as f:
        json.dump({'n_frames': np.nansum(len_arr)}, f)
    plt.plot(mean_rews)
    plt.savefig(os.path.join(exp_name, 'reward.png'))


def pandas_to_latex(df_table, latex_file, vertical_bars=False, right_align_first_column=True, header=True, index=False,
                    escape=False, multicolumn=False, **kwargs) -> None:
    """
    Function that augments pandas DataFrame.to_latex() capability.

    Args:
        df_table: dataframe
        latex_file: filename to write latex table code to
        vertical_bars: Add vertical bars to the table (note that latex's booktabs table format that pandas uses is
                          incompatible with vertical bars, so the top/mid/bottom rules are changed to hlines.
        right_align_first_column: Allows option to turn off right-aligned first column
        header: Whether or not to display the header
        index: Whether or not to display the index labels
        escape: Whether or not to escape latex commands. Set to false to pass deliberate latex commands yourself
        multicolumn: Enable better handling for multi-index column headers - adds midrules
        kwargs: additional arguments to pass through to DataFrame.to_latex()
    """
    if isinstance(df_table.index[0], tuple):
        n_col_indices = len(df_table.index[0])
    else:
        n_col_indices = 1
    n = len(df_table.columns) + n_col_indices

#   if right_align_first_column:
    cols = 'c' + 'c' * (n - 1)
#   else:
#       cols = 'r' * n

    if vertical_bars:
        # Add the vertical lines
        cols = '|' + '|'.join(cols) + '|'

    # latex = df_table.to_latex(escape=escape, index=index, column_format=cols, header=header, multicolumn=multicolumn,
    #                           **kwargs)

    # s = df_table.style.highlight_max(
    #     props='cellcolor:[HTML]{FFFF00}; color:{red}; itshape:; bfseries:;'
    # )

    s = df_table.style.format(format_vals)
    # s = s.highlight_max(axis=None, props='font-weight:bold')
    # s = df_table.style.format(format_vals)
    #                        props='cellcolor:{red}; bfseries: ;')

    latex = s.to_latex(column_format=cols, **kwargs,
        hrules=True,
        clines='skip-last;data',
        )


    with open(latex_file, 'w') as f:
        f.write(latex)


def format_vals(val, *args, **kwargs):
    if isinstance(val, float):
        return '{:,.2f}'.format(val)

    elif isinstance(val, tuple):
        # Assume first entry is mean, second is std
        return '{:,.2f} ± {:,.2f}'.format(val[0], val[1])

    elif isinstance(val, int):
        return '{:,}'.format(val)

    return val


@hydra.main(version_base="1.3", config_path="../configs", config_name="cross_eval")
def main(cfg: CrossEvalConfig):
    """This is a fake function we use to enter our hydra plugin (at 
    `hydra_plugins.cross_eval_launcher_plugin.cross_eval_launcher.CrossEvalLauncher`), from which we call `cross_eval`.
    We do this so that hydra constructs the configs corresponding to our sweep, which we then pass to `cross_eval`."""
    pass


if __name__ == '__main__':
    main()