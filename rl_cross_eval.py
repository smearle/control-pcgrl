import json
import os
import re
import csv
from pdb import set_trace as TT

import numpy as np
from matplotlib import pyplot as plt

import pandas as pd
from arguments import parse_args
from utils import get_exp_name, PROB_CONTROLS

# OVERLEAF_DIR = "/home/sme/Dropbox/Apps/Overleaf/Evolving Diverse NCA Level Generators -- AIIDE '21/tables"

# Map names of metrics recorded to the names we want to display in a table

RL_DIR = "rl_runs"

def newline(t0, t1):
    return "\\begin{tabular}[c]{@{}l@{}}" + t0 + "\\\ " + t1 + "\\end{tabular}"

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
    'net_score (mean)': 'success',
#   '(controls) net_score (mean)': newline('\\textit{control}', 'net score'),
    "diversity_score (mean)": "diversity",
#   "(controls) diversity_score (mean)": newline('\\textit{control}', 'diversity'),
    'ctrl_score (mean)': newline('control', 'success'),
#   '(controls) fixed_score (mean)': newline('\\textit{control}', 'static score'),
#   "alp_gmm": newline("ALP", "GMM"),
    "alp_gmm": newline("control", "regime"),
    "conditionals": newline("learned", " controls"),
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

    return data
    return "{}".format(data)


def flatten_stats(stats, controllable=False):
    flat_stats = {}

    def add_key_val(key, val):
        if controllable and key != "% train archive full":
#           key = "(controls) " + key
            if key in header_text:
                key = header_text[key]
            key = ', '.join([c for c in stats['controls'] if c is not None]) + '||' + key

        if "%" in key:
            val *= 100
        elif "playability" in key:
            val /= 10

        if key in header_text:
            key = header_text[key]
        flat_stats[key] = val

    for k, v in stats.items():
        if isinstance(v, dict):
            key_0 = k

            for k1, v1 in v.items():
                key = "{} ({})".format(key_0, k1)
                value = v1
                add_key_val(key, value)
        else:
            add_key_val(k, v)

    return flat_stats


def compile_results(settings_list, no_plot=False):
    batch_exp_name = settings_list[0]["experiment_id"]
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
    keys = [
            "problem", 
            "representation", 
            "conditionals", 
            "alp_gmm", 
            "change_percentage"
            ]
    columns = None
    data = []
    vals = []

    for i, settings in enumerate(settings_list):
        val_lst = []

        controllable = False
        for k in keys:
            v = settings[k]
            if k == 'conditionals':
                if k != ['NONE']:
                    controllable = True
            if isinstance(settings[k], list):
                if len(settings[k]) < 2:
                    val_lst.append(", ".join(settings[k]))
                else:
                    val_lst.append(newline(settings[k][0]+', ', v[1]))
            elif k == 'alp_gmm':
                if not controllable:
                    v = ''
                elif v:
                    v = 'ALP-GMM'
                else:
                    v = newline('uniform', 'random')
                val_lst.append(v)
            else:
                val_lst.append(v)
        args = parse_args(load_args=settings)
        arg_dict = vars(args)
        # FIXME: well this is stupid
        arg_dict["cond_metrics"] = arg_dict.pop("conditionals")
        exp_name = get_exp_name(
            arg_dict.pop("problem"), arg_dict.pop("representation"), **arg_dict
        ) + "_{}_log".format(batch_exp_name)
        # NOTE: For now, we run this locally in a special directory, to which we have copied the results of eval on
        # relevant experiments.
        exp_name = os.path.join(RL_DIR, exp_name)
        eval_dir = os.path.join(exp_name, "eval")
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
        fixLvl_stats = json.load(open(fixTrgs_stats_f, "r"))
        flat_stats = flatten_stats(fixLvl_stats)
        for stats_f in ctrl_stats_files:
            stats = json.load(open(stats_f, "r"))
            flat_stats.update(flatten_stats(stats, controllable=True))

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
    controls_id = keys.index('conditionals')
    regime_id = keys.index('alp_gmm')
    for i, tpl in enumerate(tuples):
        if tpl[controls_id] == 'NONE':
            tpl = list(tpl)
            tpl[regime_id] = '---'
            tpl = tuple(tpl)
            tuples[i] = tpl

    for k in keys:
        if k in header_text:
            new_keys.append(header_text[k])
        else:
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
    for col in columns:
        if '||' not in col:
            col_tuples.append(('fixed targets', '---', col))
        else:
            controls = col.split('||')[0]
            col_tuples.append(('controlled targets', controls, col.split('||')[-1]))
    columns = pd.MultiIndex.from_tuples(col_tuples, names=['', 'evaluated controls', ''])
    df = pd.DataFrame(data=data, index=index, columns=columns)
#   df = df.sort_values(by=new_keys, axis=0)
#   new_keys = [tuple(t) for t in tuples]
#   df = df.sort_values(by=new_keys, axis=0)

    csv_name = r"{}/cross_eval_{}.csv".format(RL_DIR, batch_exp_name)
    html_name = r"{}/cross_eval_{}.html".format(RL_DIR, batch_exp_name)
    df.to_csv(csv_name)
    df.to_html(html_name)
#   print(df)

    #   tex_name = r"{}/zelda_empty-path_cell_{}.tex".format(OVERLEAF_DIR, batch_exp_name)

#   for p in ["binary", "zelda", "sokoban"]:
    for p in ["binary"]:
        tex_name = "{}/{}_{}.tex".format(RL_DIR, p, batch_exp_name)
        try:
            df_tex = df.loc[p, "narrow"]
        except KeyError:
            continue
#       print(p)
        p_name = p + '_ctrl'
        lcl_conds = ['---'] + ['-'.join(pi) if len(pi) < 2 else newline(pi[0] + ', ', pi[1]) for pi in local_controls[p_name]]
#       print(lcl_conds)
#       df_tex = df_tex.loc[lcl_conds]
#       df_tex = df_tex.sort_values(by=['ALP GMM'])
        z_cols_fixed = [
            header_text["net_score (mean)"],
            header_text["diversity_score (mean)"],
        ]
        z_cols_ctrl = [
#           header_text["net_score (mean)"],
            header_text["ctrl_score (mean)"],
#           header_text["(controls) fixed_score (mean)"],
            header_text["diversity_score (mean)"],
        ]
        n_col_heads = len(z_cols_fixed)
        z_cols = list(zip(['fixed targets'] * n_col_heads, ['---'] * n_col_heads, z_cols_fixed))
        for ctrl_set in PROB_CONTROLS[p + '_ctrl']:
            n_col_heads = len(z_cols_ctrl)
            z_cols += list(zip(['controlled targets'] * n_col_heads, [', '.join(ctrl_set)] * n_col_heads, z_cols_ctrl))
        df_tex = df_tex[z_cols]
        #   df_tex = df.drop(columns=z_cols)
#       df_tex['fixed targets'] = df_tex['fixed targets'][z_cols[0:2]]
#       df_tex['controlled targets'] = df_tex['fixed targets'][z_cols[2:]]
        df_tex = df_tex * 100
        df_tex = df_tex.astype(float).round(0)
        dual_conds = ['---', lcl_conds[1]]
        for k in z_cols:
            print(k)
            if k in df_tex:
                print(k)
#               df_tex.loc[dual_conds][k] = df_tex.loc[dual_conds][k].apply(
#                   lambda data: bold_extreme_values(data, data_max=df_tex.loc[dual_conds][k].max())
#               )
                print(df_tex[k].max())
                df_tex[k] = df_tex[k].apply(
                    lambda data: bold_extreme_values(data, data_max=df_tex[k].max())
                )
#       df_tex = df_tex.round(2)
#       df_tex.reset_index(level=0, inplace=True)
#       print(df_tex)

#       with open(tex_name, "w") as tex_f:
#           col_widths = "p{0.5cm}p{0.5cm}p{0.5cm}p{0.5cm}p{0.5cm}p{0.5cm}p{0.8cm}p{0.8cm}p{0.8cm}"
        pandas_to_latex(
            df_tex,
#           df_tex.to_latex(
            tex_name,
            vertical_bars=True,
            index=True,
            bold_rows=True,
            header=True,
            columns=z_cols,
            multirow=True,
            multicolumn=True,
            multicolumn_format='c|',
            # column_format= 'r|' * len(index) + 'c|' * len(df_tex.columns),
            escape=False,
            caption=("Performance of controllable {}-generating agents with learning-progress-informed and uniform-random control regimes, and baseline (single-objective) agents with various change percentage allowances. Agents are tested both on a baseline task with metric targets fixed at their default values, and control tasks, in which controllable metric targets are sampled over a grid.".format(p)),
            label={"tbl:{}".format(p)},
        )

    #   # Remove duplicate row indices for readability in the csv
    #   df.reset_index(inplace=True)
    #   for k in new_keys:
    #       df.loc[df[k].duplicated(), k] = ''
    #   csv_name = r"{}/cross_eval_{}.csv".format(OVERLEAF_DIR, batch_exp_name)

def pandas_to_latex(df_table, latex_file, vertical_bars=False, right_align_first_column=True, header=True, index=False,
                    escape=False, multicolumn=False, **kwargs) -> None:
    """
    Function that augments pandas DataFrame.to_latex() capability.
    :param df_table: dataframe
    :param latex_file: filename to write latex table code to
    :param vertical_bars: Add vertical bars to the table (note that latex's booktabs table format that pandas uses is
                          incompatible with vertical bars, so the top/mid/bottom rules are changed to hlines.
    :param right_align_first_column: Allows option to turn off right-aligned first column
    :param header: Whether or not to display the header
    :param index: Whether or not to display the index labels
    :param escape: Whether or not to escape latex commands. Set to false to pass deliberate latex commands yourself
    :param multicolumn: Enable better handling for multi-index column headers - adds midrules
    :param kwargs: additional arguments to pass through to DataFrame.to_latex()
    :return: None
    """
    n = len(df_table.columns) + len(df_table.index[0])

    if right_align_first_column:
        cols = 'r' + 'c' * (n - 1)
    else:
        cols = 'c' * n

    if vertical_bars:
        # Add the vertical lines
        cols = '|' + '|'.join(cols) + '|'

    latex = df_table.to_latex(escape=escape, index=index, column_format=cols, header=header, multicolumn=multicolumn,
                              **kwargs)
    latex = latex.replace('\\begin{table}', '\\begin{table*}')
    latex = latex.replace('\end{table}', '\end{table*}')

    if vertical_bars:
        # Remove the booktabs rules since they are incompatible with vertical lines
        latex = re.sub(r'\\(top|mid|bottom)rule', r'\\hline', latex)

    # Multicolumn improvements - center level 1 headers and add midrules
    if multicolumn:
        latex = latex.replace(r'{l}', r'{c}')

        offset = len(df_table.index[0])
#       offset = 1
        midrule_str = ''
        for i, col in enumerate(df_table.columns.levels[0]):
            indices = np.nonzero(np.array(df_table.columns.codes[0]) == i)[0]
            hstart = 1 + offset + indices[0]
            hend = 1 + offset + indices[-1]
#           midrule_str += rf'\cmidrule(lr){{{hstart}-{hend}}} '
            midrule_str += rf'\cline{{{hstart}-{hend}}}'

        # Ensure that headers don't get colored by row highlighting
#       midrule_str += r'\rowcolor{white}'

        latex_lines = latex.splitlines()
        # FIXME: ad-hoc row indices here
        # TODO: get row index automatically then iterate until we get to the end of the multi-cols
        latex_lines.insert(7, midrule_str)
        latex_lines.insert(9, midrule_str)
        latex = '\n'.join(latex_lines)

    with open(latex_file, 'w') as f:
        f.write(latex)

def listolists_to_arr(a):
    b = np.empty([len(a), len(max(a, key=lambda x: len(x)))])
    b[:] = np.NaN
    for i, j in enumerate(a):
        b[i][0:len(j)] = j
    return b

def plot_csv(exp_name):
    monitor_files = [os.path.join(exp_name, f) for f in os.listdir(exp_name) if "monitor.csv" in f]
    g_rew_id, g_len_id, g_time_id = None, None, None
    g_rewards, g_lens, g_times = [], [], []
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
        g_rewards.append(rewards)
        g_lens.append(lens)
        g_times.append(times)
    if not g_rewards:
        return
    rew_arr, len_arr, time_arr = listolists_to_arr(g_rewards), listolists_to_arr(g_lens), listolists_to_arr(g_times)
    mean_rews, mean_lens, mean_times = np.nanmean(rew_arr, axis=0), np.nanmean(len_arr, axis=0), np.nanmean(time_arr, axis=0)
    plt.figure()
    mean_rews = np.convolve(mean_rews, [1/7 for i in range(7)])
    with open(os.path.join(exp_name, 'stats.json'), 'w') as f:
        json.dump({'n_frames': np.nansum(len_arr)}, f)
    plt.plot(mean_rews)
    plt.savefig(os.path.join(exp_name, 'reward.png'))
