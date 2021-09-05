from pdb import set_trace as TT
import json
import os

import pandas as pd

from evo_args import get_args, get_exp_name
from tex_formatting import pandas_to_latex, newline

# OVERLEAF_DIR = "/home/sme/Dropbox/Apps/Overleaf/Evolving Diverse NCA Level Generators -- AIIDE '21/tables"

# Attempt to make shit legible
col_keys = {
    "generations completed": "n_gen",
    "% train archive full": "coverage",
    "(generalize) % train archive full": "(infer) coverage",
    "(generalize) % elites maintained": "(infer) archive maintained",
#   "(generalize) % elites maintained": newline("(infer) archive", "maintained"),
    "(generalize) % QD score maintained": "(infer) QD score maintained",
    "(generalize) QD score": "(infer) QD score",
}

col_key_linebreaks = {
    'archive maintained': newline("archive", "maintained"),
    'QD score maintained': newline("QD score", "maintained"),
}

row_idx_names = {
    "fix_level_seeds": "latents",
    "fix_elites": "elites",
    "n_init_states": newline("n.", "latents"),
    "n_steps": newline("n.", "steps"),
}

# flatten the dictionary here


def bold_extreme_values(data, data_max=-1, col_name=None):

    if data == data_max:
        bold = True
    else: bold = False

    if "QD score" in col_name:
        data = int(data)
    if any(c in col_name for c in ["archive size",  "QD score"]):
        data = "{:,}".format(data)
    else:
        data = "{:.1f}".format(data)

    if bold:
#       data = "\\cellcolor{blue!25} "
        data = "\\bfseries {}".format(data)

    print(col_name)
    if "maintained" in col_name[1]:
        data = "{} \%".format(data)

    return data


def flatten_stats(stats, tex, evaluation=False):
    '''Process jsons saved for each experiment, replacing hierarchical dicts with a 1-level list of keys.
    args:
    - evaluation: True iff we're looking at stats when latents were randomized, False when agent is evaluated on
                      latents with which it joined the archive.
    - tex: True iff we're formatting for .tex output
    '''
    # TODO: maybe we can derive multicolumn hierarchy from this directly?
    flat_stats = {}

    def add_key_val(key, val):
        if evaluation:
            key = "(generalize) " + key

        if "%" in key:
            val *= 100
        elif "playability" in key:
            val /= 10

        if tex and key in col_keys:
            key = col_keys[key]
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


def compile_results(settings_list, tex=False):
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
    hyperparams = [
        "problem",
        "behavior_characteristics",
        "model",
        "representation",
        "n_init_states",
        "fix_level_seeds",
#       "fix_elites",
        "n_steps",
    ]

    hyperparam_rename = {
        "model" : {
            "CPPN": "VanillaCPPN",
            "GenSinCPPN": "CPPN",
        },
        "fix_level_seeds": {
            True: "Fix",
            False: "Re-sample",
        },
        "fix_elites": {
            True: "Fix",
            False: "Re-evaluate",
        },
    }
    assert len(hyperparams) == len(set(hyperparams))
    col_indices = None
    data = []
    vals = []

    for i, settings in enumerate(settings_list):
        val_lst = []

        bc_names = settings['behavior_characteristics']

        for k in hyperparams:
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
        flat_stats = flatten_stats(fixLvl_stats, tex=tex)
        flat_stats.update(flatten_stats(stats, tex=tex, evaluation=True))

        if col_indices is None:
            # grab columns (json keys) from any experiment's stats json, since they should all be the same
            col_indices = list(flat_stats.keys())

        for j, c in enumerate(col_indices):
            if c not in flat_stats:
                data[-1].append("N/A")
            else:
                data[-1].append(flat_stats[c])

    tuples = vals
    for i, tpl in enumerate(tuples):
        # Preprocess row headers
        for j, hyper_val in enumerate(tpl):
            hyper_name = hyperparams[j]
            if hyper_name in hyperparam_rename:
                if hyper_val in hyperparam_rename[hyper_name]:
                    tpl = list(tpl)
                    tpl[j] = hyperparam_rename[hyper_name][hyper_val]
            tpl = tuple(tpl)
        tuples[i] = tpl


    # Rename headers
    new_keys = []



    for k in hyperparams:
        if tex and k in col_keys:
            new_keys.append(col_keys[k])
        elif k not in new_keys:
            new_keys.append(k)
        else:
            pass
#           new_keys.append('{}_{}'.format(k, 2))
    print(tuples, new_keys)
    row_indices = pd.MultiIndex.from_tuples(tuples, names=new_keys)
    #   df = index.sort_values().to_frame(index=True)
    z_cols = [
#       "% train archive full",
        "archive size",
        "QD score",
#       "(generalize) % train archive full",
        "(generalize) archive size",
        "(generalize) QD score",
        "(generalize) archive maintained",
        "(infer) QD score maintained",
    ]
    z_cols = [col_keys[z] if z in col_keys else z for z in z_cols]
    # Hierarchical columns!
    def hierarchicalize_col(col):
        if col.startswith('(infer)'):
            return ('Evaluation', ' '.join(col.split(' ')[1:]))
            # return ('Evaluation', col)
        elif col.startswith('(generalize)'):
            # return ('Generalization', col.strip('(generalize)'))
            return ('Evaluation', ' '.join(col.split(' ')[1:]))
        else:
            return ('Training', col)
    for i, col in enumerate(z_cols):
        hier_col = hierarchicalize_col(col)
        z_cols[i] = tuple([col_key_linebreaks[hier_col[i]] if hier_col[i] in col_key_linebreaks else hier_col[i] for
                           i in range(len(hier_col))])
    col_tuples = []
    for col in col_indices:
        hier_col = hierarchicalize_col(col)
        col_tuples.append(tuple([col_key_linebreaks[hier_col[i]] if hier_col[i] in col_key_linebreaks else hier_col[i] for
                                 i in range(len(hier_col))]))
        # columns = pd.MultiIndex.from_tuples(col_tuples, names=['', newline('evaluated', 'controls'), ''])
    col_indices = pd.MultiIndex.from_tuples(col_tuples)
    # columns = pd.MultiIndex.from_tuples(col_tuples)
    df = pd.DataFrame(data=data, index=row_indices, columns=col_indices).sort_values(by=new_keys)

    csv_name = r"{}/cross_eval_{}.csv".format(EVO_DIR, batch_exp_name)
    html_name = r"{}/cross_eval_{}.html".format(EVO_DIR, batch_exp_name)
    df.to_html(html_name)
    print(df)
    for i, k in enumerate(new_keys):
        if k in row_idx_names:
            new_keys[i] = row_idx_names[k]
    df.index.rename(new_keys, inplace=True)
#   df.rename(col_keys, axis=1)

    df.to_csv(csv_name)

    if not tex:
        return

#   tex_name = r"{}/zelda_empty-path_cell_{}.tex".format(OVERLEAF_DIR, batch_exp_name)
    tex_name = r"{}/cross_eval_{}.tex".format(EVO_DIR, batch_exp_name)
    df = df.round(1)
    df_tex = df.loc["binary_ctrl", "symmetry-path-length", :, "cellular"].round(1)
#   df_tex = df.loc["zelda_ctrl", "nearest-enemy-path-length", :, "cellular"].round(1)

    for k in z_cols:
        if k in df_tex:
            df_tex[k] = df_tex[k].apply(
                lambda data: bold_extreme_values(data, data_max=df_tex[k].max(), col_name=k)
            )
    df_tex = df_tex.round(1)
    df.reset_index(level=0, inplace=True)
    print(df_tex)
    col_widths = "p{0.5cm}p{0.5cm}p{0.5cm}p{0.8cm}p{0.8cm}p{0.8cm}p{0.8cm}"
    print('Col names:')
    [print(z) for z in df_tex.columns]
    print('z_col names:')
    [print(z) for z in z_cols]
    pandas_to_latex(
        df_tex,
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
##      caption=(
##          "Zelda, with emptiness and path-length as measures. Evolution runs in which agents are exposed to more random seeds appear to generalize better during inference. Re-evaluation of elites on new random seeds during evolution increases generalizability but the resulting instability greatly diminishes CMA-ME's ability to meaningfully explore the space of generators. All experiments were run for 10,000 generations"
##      ),
        label={'tbl:zelda_empty-path_cell_{}'.format(batch_exp_name)},
        bold_rows=True,
    )


#   # Remove duplicate row indices for readability in the csv
#   df.reset_index(inplace=True)
#   for k in new_keys:
#       df.loc[df[k].duplicated(), k] = ''
#   csv_name = r"{}/cross_eval_{}.csv".format(OVERLEAF_DIR, batch_exp_name)

