import math
from pdb import set_trace as TT
import json
import os

import numpy as np
import matplotlib
import pandas as pd
import pingouin as pg
import scipy.stats

from evo.args import get_args, get_exp_dir, get_exp_name
from tex_formatting import align, newline, pandas_to_latex
from matplotlib import pyplot as plt

# OVERLEAF_DIR = "/home/sme/Dropbox/Apps/Overleaf/Evolving Diverse NCA Level Generators -- AIIDE '21/tables"

# Attempt to make shit legible
col_keys = {
    "generations completed": "n_gen",
    "% train archive full": "coverage",
    "(generalize) % train archive full": "(infer) coverage",
    "(generalize) % elites maintained": "(infer) archive maintained",
#   "(generalize) % elites maintained": newline("(infer) archive", "maintained"),
    "(generalize) % QD score maintained": "(infer) QD score maintained",
    "(generalize) eval QD score": "(infer) QD score",
#   "(generalize) eval QD score": "(infer) qd_score",
    "(generalize) QD score": "QD score 2",
    "(generalize) diversity (mean)": "(infer) diversity",
    "diversity (mean)": "diversity",
}

col_key_linebreaks = {
    'archive maintained': newline("archive", "maintained"),
    'QD score maintained': newline("QD score", "maintained"),
    # 'QD score': align('QD score', "c"),
    'diversity': newline("generator", "diversity", align="r"),
}

row_idx_names = {
    "exp_name": "exp_name",
    "algo": "algorithm",
    "fix_level_seeds": "latents",
    "fix_elites": "elites",
    "n_init_states": newline("batch", "size"),
    "n_steps": newline("num.", "steps"),
    "step_size": "step size",
}

# flatten the dictionary here


def bold_extreme_values(data, data_max=-1, col_name=None):

    data, err = data
    if data == err == 0.0:
        return "---"
        
    if data == data_max:
        bold = True
    else: bold = False

    if "QD score" in col_name:
        # FIXME ad hoc
        if np.isnan(data):
            data = np.nan
        else:
            data = int(data / 10000)
        if np.isnan(err):
            err = np.nan
        else:
            err = err / 10000
        # data, err = int(data / 10000), int(err / 10000)
    if any(c in col_name for c in ["archive size",  "QD score"]):
        data = "{:,.0f}".format(data)
    elif "diversity" in col_name[1]:
        data = "{:.2f}".format(data)
    else:
        data = "{:.1f}".format(data)

    print(col_name)
    if "maintained" in col_name[1]:
        data = "{} \%".format(data)

    if False and np.any(['diversity' in c for c in col_name]):
        err = "{:.1e}".format(err)
    else:
        err = "{:.1f}".format(err)

    if bold:
        data = '\\textbf{' + str(data) + '} ± ' + str(err)
    else:
        data = f'{data} ± {err}'
    return data


def flatten_stats(stats, tex, evaluation=False):
    '''Process jsons saved for each experiment, replacing hierarchical dicts with a 1-level list of keys.

    Args:
        evaluation (bool): True iff we're looking at stats when latents were randomized, False when agent is evaluated on
            latents with which it joined the archive.
        tex (bool): True iff we're formatting for .tex output
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
#   batch_exp_name = settings_list[0]["exp_name"]
    EVO_DIR = "evo_runs"
    EVAL_DIR = "eval_experiment"
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
#       "problem",
#       "behavior_characteristics",
        "model",
        "algo",
        "step_size",
#       "representation",
        "n_init_states",
#       "fix_level_seeds",
#       "fix_elites",
        "n_steps",
        "exp_name",
    ]

    hyperparam_rename = {
        "model" : {
            # "AuxNCA": newline("NCA with", "auxiliary channels", "r"),
            # "AuxNCA": "NCA with auxiliary channels",
            # "Sin2CPPN": newline("fixed-topology", "CPPN", "r"),
            "Sin2CPPN": "fixed-topology CPPN",
            # "GenSin2CPPN2": newline("generative,", "fixed-topology CPPN", "r"),
            "GenSin2CPPN2": "generative, fixed-topology CPPN",
            "GenCPPN2": "generative CPPN",
            # "CPPN": "Vanilla CPPN",
            # "GenSinCPPN": " "+newline("Fixed", "CPPN"),
            # "GenSinCPPN": " Fixed CPPN",
            # "GenCPPN": "CPPN",
        },
        "algo" : {
            "CMAME": "CMA-ME",
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
    row_tpls = []

    for i, settings in enumerate(settings_list):
        val_lst = []

        bc_names = settings['behavior_characteristics']

        for k in hyperparams:
            if isinstance(settings[k], list):
                val_lst.append("-".join(settings[k]))
            else:
                val_lst.append(settings[k])
        args, arg_dict = get_args(load_args=settings)
        exp_dir = get_exp_dir(get_exp_name(args, arg_dict))
        # NOTE: For now, we run this locally in a special directory, to which we have copied the results of eval on
        # relevant experiments.
#       exp_name = exp_name.replace("evo_runs/", "{}/".format(EVO_DIR))
        # stats_f = os.path.join(exp_name, "train_time_stats.json")
        stats_f = os.path.join(exp_dir, "stats.json")
        fix_lvl_stats_f = os.path.join(exp_dir, "statsfixLvls.json")

        if not (os.path.isfile(stats_f) and os.path.isfile(fix_lvl_stats_f)):
            print("skipping evaluation of experiment due to missing stats file(s): {}".format(exp_dir))
            continue
        row_tpls.append(tuple(val_lst))
        data.append([])
        stats = json.load(open(stats_f, "r"))
        print(fix_lvl_stats_f)
        fix_lvl_stats = json.load(open(fix_lvl_stats_f, "r"))
        flat_stats = flatten_stats(fix_lvl_stats, tex=tex)
        flat_stats.update(flatten_stats(stats, tex=tex, evaluation=True))

        if col_indices is None:
            # grab columns (json keys) from any experiment's stats json, since they should all be the same
            col_indices = list(flat_stats.keys())

        generative = settings['n_init_states'] != 0

        for j, c in enumerate(col_indices):
            if c not in flat_stats:
                data[-1].append("N/A")
            # All values that are not applicable for non-generative (indirect encoding) models are zeroed out.
            if not generative and (c == 'diversity' or c.startswith('(infer)') or c.startswith('(generalize)')):
                data[-1].append(0)
            else:
                data[-1].append(flat_stats[c])

    def analyze_metric(metric):
        """Run statistical significance tests for come metric (i.e. column header) of interest."""
        # Do one-way anova test over models. We need a version of the dataframe with "model" as a column (I guess).
        # NOTE: This is only meaningful when considering a batch of experiments varying only over 1 hyperparameter
        qd_score_idx = col_indices.index(metric)
        oneway_anova_data = {'model': [v[0] for v in row_tpls], metric: [d[qd_score_idx] for d in data]}
        oneway_anova_df = pd.DataFrame(oneway_anova_data)
        oneway_anova = pg.anova(data=oneway_anova_df, dv=metric, between='model', detailed=True)
        oneway_anova.to_latex(os.path.join('eval_experiment', f'oneway_anova_{metric}.tex'))
        oneway_anova.to_html(os.path.join('eval_experiment', f'oneway_anova_{metric}.html'))

        pairwise_tukey = pg.pairwise_tukey(data=oneway_anova_df, dv=metric, between='model')
        pairwise_tukey.to_latex(os.path.join('eval_experiment', f'pairwise_tukey_{metric}.tex'))
        pairwise_tukey.to_html(os.path.join('eval_experiment', f'pairwise_tukey_{metric}.html'))

    # for metric in ['archive size', 'QD score', '(infer) QD score', '(generalize) archive size', '(infer) diversity', 'diversity']:
        # analyze_metric(metric)

    # Rename hyperparameter names (row indices)
    new_keys = []
    for k in hyperparams:
        if tex and k in col_keys:
            new_keys.append(col_keys[k])
        elif k not in new_keys:
            new_keys.append(k)
        else:
            pass
#           new_keys.append('{}_{}'.format(k, 2))

    # Sort rows (of hyperparam values and corresponding data) manually.
    def sort_rows(row_tpl, row_keys):
        model_name = row_tpl[row_keys.index('model')]
        return ["CPPN", "Sin2CPPN", "GenCPPN2", "GenSin2CPPN2", "Decoder", "NCA", "AuxNCA"].index(model_name)
    rows_data_sorted = sorted(zip(row_tpls, data), key=lambda x: sort_rows(x[0], new_keys))
    row_tpls, data = [list(e) for e in zip(*rows_data_sorted)]

    # Preprocess/rename hyperparameter values (row index headers)
    for i, tpl in enumerate(row_tpls):
        for j, hyper_val in enumerate(tpl):
            hyper_name = hyperparams[j]
            if hyper_name in hyperparam_rename:
                if hyper_val in hyperparam_rename[hyper_name]:
                    tpl = list(tpl)
                    tpl[j] = hyperparam_rename[hyper_name][hyper_val]
            tpl = tuple(tpl)
        row_tpls[i] = tpl

    row_indices = pd.MultiIndex.from_tuples(row_tpls, names=new_keys)
    #   df = index.sort_values().to_frame(index=True)
    z_cols = [
#       "% train archive full",
        "archive size",
        "QD score",
        "diversity (mean)",
#       "(generalize) % train archive full",
        "(generalize) archive size",
        "(generalize) eval QD score",
#       "(generalize) archive maintained",
#       "(infer) QD score maintained",
        "(generalize) diversity (mean)"
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
#       z_cols[i] = tuple([hier_col[i] for i in range(len(hier_col))])
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
    df = pd.DataFrame(data=data, index=row_indices, columns=col_indices)

    csv_name = r"{}/cross_eval_multi.csv".format(EVAL_DIR)
    html_name = r"{}/cross_eval_multi.html".format(EVAL_DIR)
    print(df)
    for i, k in enumerate(new_keys):
        if k in row_idx_names:
            new_keys[i] = row_idx_names[k]

    df.index.rename(new_keys, inplace=True)
    # Take mean/std over multiple runs with the same relevant hyperparameters
    new_rows = []
    new_row_names = []
    # Get some p-values for statistical significance
    eval_qd_scores = []
    for i in range(df.shape[0]):
        row = df.iloc[i]
        name = row.name
        if name[:-1] in new_row_names:
            continue
        new_row_names.append(name[:-1])
        repeat_exps = df.loc[name[:-1]]
        eval_qd_scores.append(repeat_exps[('Evaluation', 'QD score')].to_numpy())
        mean_exp = repeat_exps.mean(axis=0)
        std_exp = repeat_exps.std(axis=0)
        mean_exp = [(i, e) for i, e in zip(mean_exp, std_exp)]
        new_rows.append(mean_exp)
    pvals = np.zeros(shape=(len(new_row_names), len(new_row_names)))

    for i, vi in enumerate(eval_qd_scores):
        for j, vj in enumerate(eval_qd_scores):
            if vi.sum() == vj.sum() == 0:
                pvals[i,j] = 1.0
                continue
            pvals[i,j] = scipy.stats.mannwhitneyu(vi, vj)[1]

    im = plt.figure()

    cross_eval_heatmap(pvals, row_labels=new_row_names, col_labels=new_row_names, title='Eval QD score p-values', pvals=True, swap_xticks=False)
#   plt.xticks(range(len(new_row_names)), labels=[str(i) for i in new_row_names], rotation='vertical')
    df.to_csv(csv_name)
    df.to_html(html_name)

    # Create new data-frame that squashes different iterations of the same experiment
    csv_name = r"{}/cross_eval.csv".format(EVAL_DIR)
    html_name = r"{}/cross_eval.html".format(EVAL_DIR)
    ndf = pd.DataFrame()
    ndf = ndf.append(new_rows)
    new_col_indices = pd.MultiIndex.from_tuples(df.columns)
    new_row_indices = pd.MultiIndex.from_tuples(new_row_names)
    ndf.index = new_row_indices
    ndf.columns = new_col_indices
    ndf.index.names = df.index.names[:-1]
    ndf = ndf
    ndf.to_csv(csv_name)
    ndf.to_html(html_name)
    df.rename(col_keys, axis=1)

    df = ndf

    if not tex:
        return

    tex_name = r"{}/cross_eval.tex".format(EVAL_DIR)
#   df_tex = df.loc["binary_ctrl", "symmetry-path-length", :, "cellular"]
    df_tex = df
#   df_tex = df.loc["binary_ctrl", "symmetry-path-length", :, "cellular"].round(1)
#   df_tex = df.loc["zelda_ctrl", "nearest-enemy-path-length", :, "cellular"].round(1)

    # format the value in each cell to a string
    for k in z_cols:
        if k in df_tex:
            df_tex[k] = df_tex[k].apply(
                lambda data: bold_extreme_values(data, data_max=max([d[0] for d in df_tex[k]]), col_name=k)
            )
    df_tex = df_tex.round(1)
#   df.reset_index(level=0, inplace=True)
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
        right_align_first_column=False,
        multirow=True,
        multicolumn=True,
        multicolumn_format='c|',
        escape=False,
##      caption=(
##          "Zelda, with emptiness and path-length as measures. Evolution runs in which agents are exposed to more random seeds appear to generalize better during inference. Re-evaluation of elites on new random seeds during evo increases generalizability but the resulting instability greatly diminishes CMA-ME's ability to meaningfully explore the space of generators. All experiments were run for 10,000 generations"
##      ),
        label={'tbl:cross_eval'},
        bold_rows=True,
    )


#   # Remove duplicate row indices for readability in the csv
#   df.reset_index(inplace=True)
#   for k in new_keys:
#       df.loc[df[k].duplicated(), k] = ''
#   csv_name = r"{}/cross_eval_{}.csv".format(OVERLEAF_DIR, batch_exp_name)


### HEATMAP VISUALISATION STUFF ###

def cross_eval_heatmap(data, row_labels, col_labels, title, cbarlabel='', errors=None, pvals=False, figshape=(30,30),
                       xlabel='maps', ylabel='models', filename=None, swap_xticks=True):
   if filename is None:
      filename = title
   fig, ax = plt.subplots()
   # Remove empty rows and columns
   i = 0

   # Remove empty rows and columns
   for data_row in data:
      if np.isnan(data_row).all():
         data = np.vstack((data[:i], data[i+1:]))
         assert np.isnan(errors[i]).all()
         errors = np.vstack((errors[:i], errors[i+1:]))
         row_labels = row_labels[:i] + row_labels[i+1:]
         continue
      i += 1
   i = 0
   for data_col in data.T:
      if np.isnan(data_col).all():
         data = (np.vstack((data.T[:i], data.T[i + 1:]))).T
         assert np.isnan(errors.T[i]).all()
         errors = (np.vstack((errors.T[:i], errors.T[i+1:]))).T
         col_labels = col_labels[:i] + col_labels[i+1:]
         continue
      i += 1

#  fig.set_figheight(1.5*len(col_labels))
#  fig.set_figwidth(1.0*len(row_labels))
   fig.set_figwidth(figshape[0])
   fig.set_figheight(figshape[1])

   if pvals:
      cmap="viridis"
   else:
      cmap="magma"
   im, cbar = heatmap(data, row_labels, col_labels, ax=ax,
                      cmap=cmap, cbarlabel=cbarlabel)
   if not swap_xticks:
      im.axes.xaxis.tick_bottom()

   class CellFormatter(object):
      def __init__(self, errors):
         self.errors = errors
      def func(self, x, pos):
        #if np.isnan(x) or np.isnan(errors[pos]):
#       #   print(x, errors[pos])

        #   # Turns out the data entry is "masked" while the error entry is nan
#       #   assert np.isnan(x) and np.isnan(errors[pos])
#       #   if not np.isnan(x) and np.isnan(errors[pos]):
        #   return '--'
         if not pvals:
            x_str = "{:.1f}".format(x)
         else:
            x_str = "{:.4e}".format(x)
#           x_str = "{:.3f}".format(x)
         if errors is None:
            return x_str
         err = errors[pos]
         x_str = x_str + "  ± {:.1f}".format(err)
         return x_str
   cf = CellFormatter(errors)

   if pvals:
      textcolors = ("white", "black")
   else:
      textcolors = ("white", "black")
   texts = annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(cf.func), textcolors=textcolors)
   ax.set_title(title)

#  fig.tight_layout(rect=[1,0,1,0])
   fig.tight_layout(pad=3)
#  plt.show()
   ax.set_xlabel(xlabel)
   ax.set_ylabel(ylabel)
   plt.savefig(os.path.join(
       'eval_experiment',
      '{}.png'.format(filename),
   ))
   plt.close()


def annotate_heatmap(im, data=None, valfmt=lambda x: x,
                     textcolors=("white", "black"),
                     threshold=None, **textkw):
   """
   A function to annotate a heatmap.

   Parameters
   ----------
   im
       The AxesImage to be labeled.
   data
       Data used to annotate.  If None, the image's data is used.  Optional.
   valfmt
       The format of the annotations inside the heatmap.  This should either
       use the string format method, e.g. "$ {x:.2f}", or be a
       `matplotlib.ticker.Formatter`.  Optional.
   textcolors
       A pair of colors.  The first is used for values below a threshold,
       the second for those above.  Optional.
   threshold
       Value in data units according to which the colors from textcolors are
       applied.  If None (the default) uses the middle of the colormap as
       separation.  Optional.
   **kwargs
       All other arguments are forwarded to each call to `text` used to create
       the text labels.
   """

   if not isinstance(data, (list, np.ndarray)):
      data = im.get_array()

   # Normalize the threshold to the images color range.
   if threshold is not None:
      threshold = im.norm(threshold)
   else:
      threshold = im.norm(data.max()) / 2.

   # Set default alignment to center, but allow it to be
   # overwritten by textkw.
   kw = dict(horizontalalignment="center",
             verticalalignment="center")
   kw.update(textkw)

   # Get the formatter in case a string is supplied
   if isinstance(valfmt, str):
      valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

   # Loop over the data and create a `Text` for each "pixel".
   # Change the text's color depending on the data.
   texts = []
   for i in range(data.shape[0]):
      for j in range(data.shape[1]):
         kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
         text = im.axes.text(j, i, valfmt(data[i, j], pos=(i, j)), **kw)
         texts.append(text)

   return texts


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
   """
   Create a heatmap from a numpy array and two lists of labels.

   Parameters
   ----------
   data
       A 2D numpy array of shape (N, M).
   row_labels
       A list or array of length N with the labels for the rows.
   col_labels
       A list or array of length M with the labels for the columns.
   ax
       A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
       not provided, use current axes or create a new one.  Optional.
   cbar_kw
       A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
   cbarlabel
       The label for the colorbar.  Optional.
   **kwargs
       All other arguments are forwarded to `imshow`.
   """

   if not ax:
      ax = plt.gca()

   # Plot the heatmap
   im = ax.imshow(data, **kwargs)

   # Create colorbar
   cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
   cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

   # We want to show all ticks...
   ax.set_xticks(np.arange(data.shape[1]))
   ax.set_yticks(np.arange(data.shape[0]))
   # ... and label them with the respective list entries.
   ax.set_xticklabels(col_labels)
   plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
            rotation_mode="anchor")
   ax.set_yticklabels(row_labels)

   # Let the horizontal axes labeling appear on top.
   ax.tick_params(top=True, bottom=False,
                  labeltop=True, labelbottom=False)

   # Rotate the tick labels and set their alignment.
   plt.setp(ax.get_yticklabels(), rotation=30, ha="right",
            rotation_mode="anchor")

   # Turn spines off and create white grid.
   #ax.spines[:].set_visible(False)

   ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
   ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
   ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
   ax.tick_params(which="minor", bottom=False, left=False)

   return im, cbar