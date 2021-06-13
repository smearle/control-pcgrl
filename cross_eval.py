import json
import os
from pdb import set_trace as TT

import numpy as np
import pandas as pd

from evo_args import get_args, get_exp_name

EVO_DIR = 'evo_runs_06-12'

# flatten the dictionary here
def flatten_stats(stats, generalization=False):
    flat_stats = {}

    def add_key_val(key, val):
        if generalization and key != "% train archive full":
            key = '(generalize) ' + key
        if "%" in key:
            val *= 100
        elif "playability" in key:
            val /= 10
        flat_stats[key] = val

    for k, v in stats.items():
        if isinstance(v, dict):
            key_0 = k
            for k1, v1 in v.items():
                key = '{} ({})'.format(key_0, k1)
                value = v1
                add_key_val(key, value)
        else:
            add_key_val(k, v)
    return flat_stats


def compile_results(settings_list):
    batch_exp_name = settings_list[0]["exp_name"]
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
        "n_init_states",
        "fix_level_seeds",
        "fix_elites",
        "n_steps",
    ]
    columns = None
    data = []
    vals = []

    for i, settings in enumerate(settings_list):
        val_lst = [
            settings[k] if not isinstance(settings[k], list) else "-".join(settings[k])
            for k in keys
        ]
        args, arg_dict = get_args(load_args=settings)
        exp_name = get_exp_name(args, arg_dict)
        # NOTE: For now, we run this locally in a special directory, to which we have copied the results of eval on
        # relevant experiments.
        exp_name = exp_name.replace("evo_runs/", "{}/".format(EVO_DIR))
        stats_f = os.path.join(exp_name, "stats.json")
        fixLvl_stats_f = os.path.join(exp_name, "statsfixLvls.json")
        if not (os.path.isfile(stats_f) and os.path.isfile(fixLvl_stats_f)):
            continue
        vals.append(tuple(val_lst))
        data.append([])
        stats = json.load(open(stats_f, "r"))
        fixLvl_stats = json.load(open(fixLvl_stats_f, "r"))
        flat_stats = flatten_stats(fixLvl_stats)
        flat_stats.update(flatten_stats(stats, generalization=True))
        if columns is None:
            columns = list(flat_stats.keys())
        for j, c in enumerate(columns):
            data[-1].append(flat_stats[c])

    tuples = vals
    index = pd.MultiIndex.from_tuples(tuples, names=keys)
#   df = index.sort_values().to_frame(index=True)
    df = pd.DataFrame(data=data, index=index, columns=columns).sort_values(by=keys)
#   print(index)
    print(df)
    csv_name = r'{}/cross_eval_{}.csv'.format(EVO_DIR, batch_exp_name)
    df.to_csv(csv_name)

