import json
import os
from pdb import set_trace as TT

import numpy as np
import pandas as pd

from evo_args import get_args, get_exp_name
from tex_formatting import pandas_to_latex, newline

OVERLEAF_DIR = "/home/sme/Dropbox/Apps/Overleaf/Evolving Diverse NCA Level Generators -- AIIDE '21/tables"

# Attempt to make shit legible
header_text = {
    "fix_level_seeds": "Seeds",
    "fix_elites": "Elites",
    "generations completed": "n_gen",
    "n_init_states": newline("Num.", "seeds"),
    "n_steps": "Num. steps",
    "% train archive full": "coverage (\%)",
    "(generalize) % fresh train archive full": "(infer) coverage (\%)",
    "(generalize) % elites maintained": "(infer) maintained (\%)",
}

# flatten the dictionary here


def bold_extreme_values(data, data_max=-1):

    if data == data_max:
        return "\\bfseries {:.1f}".format(data)

    else:
        return "{:.1f}".format(data)

    return data


def flatten_stats(stats, generalization=False):
    '''Process jsons saved for each experiment, replacing hierarchical dicts with a 1-level list of keys'''
    # TODO: maybe we can derive multicolumn hierarchy from this directly?
    flat_stats = {}

    def add_key_val(key, val):
        if generalization and key != "% train archive full":
            key = "(generalize) " + key

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


def compile_results(settings_list):
    batch_exp_name = settings_list[0]["exp_name"]
    EVO_DIR = "evo_runs"
#   if batch_exp_name == "0":
#       EVO_DIR = "evo_runs_06-12"
#   else:
#       #       EVO_DIR = "evo_runs_06-13"
#       EVO_DIR = "evo_runs_06-14"
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
        "behavior_characteristics",
        "representation",
        "model",
        "n_init_states",
        "fix_level_seeds",
        "fix_elites",
        "n_steps",
    ]
    columns = None
    data = []
    vals = []

    for i, settings in enumerate(settings_list):
        val_lst = []

        for k in keys:
            if isinstance(settings[k], list):
                val_lst.append("-".join(settings[k]))
            else:
                val_lst.append(settings[k])
        args, arg_dict = get_args(load_args=settings)
        exp_name = get_exp_name(args, arg_dict)
        # NOTE: For now, we run this locally in a special directory, to which we have copied the results of eval on
        # relevant experiments.
#       exp_name = exp_name.replace("evo_runs/", "{}/".format(EVO_DIR))
        stats_f = os.path.join(exp_name, "stats.json")
        fixLvl_stats_f = os.path.join(exp_name, "statsfixLvls.json")

        if not (os.path.isfile(stats_f) and os.path.isfile(fixLvl_stats_f)):
            print("skipping evaluation of experiment due to missing stats file(s): {}".format(exp_name))
            continue
        vals.append(tuple(val_lst))
        data.append([])
        stats = json.load(open(stats_f, "r"))
        fixLvl_stats = json.load(open(fixLvl_stats_f, "r"))
        flat_stats = flatten_stats(fixLvl_stats)
        flat_stats.update(flatten_stats(stats, generalization=True))

        if columns is None:
            # grab columns (json keys) from any experiment's stats json, since they should all be the same
            columns = list(flat_stats.keys())

        for j, c in enumerate(columns):
            if c not in flat_stats:
                data[-1].append("N/A")
            else:
                data[-1].append(flat_stats[c])

    tuples = vals
    for i, tpl in enumerate(tuples):
        # Preprocess row headers
        for j, row_header in enumerate(tpl):
            if keys[j] == 'fix_level_seeds':
                tpl = list(tpl)
                if row_header == True:
                    tpl[j] = 'Fix'
                elif row_header == False:
                    tpl[j] = 'Re-sample'
            if keys[j] == 'fix_elites':
                tpl = list(tpl)
                if row_header == True:
                    tpl[j] = 'Fix'
                elif row_header == False:
                    tpl[j] = 'Re-evaluate'
            tpl = tuple(tpl)
        tuples[i] = tpl

    # Rename headers
    new_keys = []



    for k in keys:
        if k in header_text:
            new_keys.append(header_text[k])
        else:
            new_keys.append(k)
    index = pd.MultiIndex.from_tuples(tuples, names=new_keys)
    #   df = index.sort_values().to_frame(index=True)
    z_cols = [header_text["% train archive full"], header_text["(generalize) % fresh train archive full"], header_text["(generalize) % elites maintained"]]
    # Hierarchical columns!
    def hierarchicalize_col(col):
        if col.startswith('(infer)'):
            return ('Evaluation', ' '.join(col.split(' ')[1:]))
            # return ('Evaluation', col)
        elif col.startswith('(generalize)'):
            # return ('Generalization', col.strip('(generalize)'))
            return ('Generalization', col)
        else:
            return ('Training', col)
    for i, col in enumerate(z_cols):
        z_cols[i] = hierarchicalize_col(col)
    col_tuples = []
    for col in columns:
        col_tuples.append(hierarchicalize_col(col))
    # columns = pd.MultiIndex.from_tuples(col_tuples, names=['', newline('evaluated', 'controls'), ''])
    columns = pd.MultiIndex.from_tuples(col_tuples)
    # columns = pd.MultiIndex.from_tuples(col_tuples)
    df = pd.DataFrame(data=data, index=index, columns=columns).sort_values(by=new_keys)
    #   print(index)

    csv_name = r"{}/cross_eval_{}.csv".format(EVO_DIR, batch_exp_name)
    html_name = r"{}/cross_eval_{}.html".format(EVO_DIR, batch_exp_name)
    df.to_csv(csv_name)
    df.to_html(html_name)
    print(df)

#   tex_name = r"{}/zelda_empty-path_cell_{}.tex".format(OVERLEAF_DIR, batch_exp_name)
    tex_name = r"{}/zelda_empty-path_cell_{}.tex".format(EVO_DIR, batch_exp_name)
    df = df.round(1)
    df_zelda = df.loc["zelda_ctrl", "emptiness-path-length", "cellular"].round(1)

    for k in z_cols:
        if k in df_zelda:
            df_zelda[k] = df_zelda[k].apply(
                lambda data: bold_extreme_values(data, data_max=df_zelda[k].max())
            )
    df_zelda = df_zelda.round(1)
    df.reset_index(level=0, inplace=True)
    print(df_zelda)
    col_widths = "p{0.5cm}p{0.5cm}p{0.5cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}"
    pandas_to_latex(
        df_zelda,
        tex_name,
        index=True,
        header=True,
        vertical_bars=True,
        columns=z_cols,
        # column_format=col_widths,
        multirow=True,
        multicolumn=True,
        multicolumn_format='c|',
        escape=False,
        caption=(
            "Zelda, with emptiness and path-length as measures. Evolution runs in which agents are exposed to more random seeds appear to generalize better during inference. Re-evaluation of elites on new random seeds during evolution increases generalizability but the resulting instability greatly diminishes CMA-ME's ability to meaningfully explore the space of generators. All experiments were run for 10,000 generations"
        ),
        label={'tbl:zelda_empty-path_cell_{}'.format(batch_exp_name)},
        bold_rows=True,
    )


#   # Remove duplicate row indices for readability in the csv
#   df.reset_index(inplace=True)
#   for k in new_keys:
#       df.loc[df[k].duplicated(), k] = ''
#   csv_name = r"{}/cross_eval_{}.csv".format(OVERLEAF_DIR, batch_exp_name)

