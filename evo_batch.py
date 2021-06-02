'''
Launch a batch of experiments on a SLURM cluster.

WARNING: This will kill all ray processes running on the current node after each experiment, to avoid memory issues from dead processes.
'''
import os
import copy
import json
import re
import argparse

problems = [
    'binary_ctrl',
#   'zelda_ctrl',
#   'sokoban_ctrl',
#   'smb_ctrl',
    ]
representations = [
#   'cellular',
    'wide',
#   'narrow',
#   'turtle',
    ]
global_bcs = [
#   ['NONE'],
#   ['emptiness', 'symmetry'],
]
local_bcs = {
    'binary_ctrl': [
#       ['regions', 'path-length'],
#       ['emptiness', 'path-length'],
        ['symmetry', 'path-length'],
    ],
    'zelda_ctrl': [
#       ['nearest-enemy', 'path-length'],
        ['emptiness', 'path-length'],
        ['symmetry', 'path-length'],
    ],
    'sokoban_ctrl': [
#       ['crate', 'sol-length'],
        ['emptiness', 'sol-length'],
        ['symmetry', 'sol-length'],
    ],
    'smb_ctrl': [
#       ['enemies', 'jumps'],
        ['emptiness', 'jumps'],
        ['symmetry', 'jumps'],
    ],
}
models = [
#   'NCA',
    'CNN',  # Doesn't learn atm
]

# Reevaluate elites on new random seeds after inserting into the archive?
fix_elites = [
    True,
    False,
]

# Fix a set of random levels with which to seed the generator, or use new ones each generation?
fix_seeds = [
    True,
    False,
]

def launch_batch(exp_name):
    if TEST:
        print('Testing locally.')
    else:
        print('Launching batch of experiments on SLURM.')
    with open('configs/evo/default_settings.json', 'r') as f:
        default_config = json.load(f)
    print('Loaded default config:\n{}'.format(default_config))

    if TEST:
        default_config['n_generations'] = 1
    i = 0

    for prob in problems:
        prob_bcs = global_bcs + local_bcs[prob]

        for rep in representations:
            for model in models:

                if model == 'CNN' and rep == 'cellular':
                    # This would necessitate an explosive number of model params so we'll not run it
                    continue

                for bc_pair in prob_bcs:

                    for fix_el in fix_elites:
                        for fix_seed in fix_seeds:

                            # No reason to re-evaluate other than random seeds so this would cause an error
                            if fix_seed and not fix_el:
                                continue

                            # Edit the sbatch file to load the correct config file
                            with open('evo_train.sh', 'r') as f:
                                content = f.read()
                                new_content = re.sub(
                                    'python evolve.py -la \d+',
                                    'python evolve.py -la {}'.format(i), content)
                            with open('evo_train.sh', 'w') as f:
                                f.write(new_content)
                            # Write the config file with the desired settings
                            exp_config = copy.deepcopy(default_config)
                            exp_config.update({
                                'problem': prob,
                                'representation': rep,
                                'behavior_characteristics': bc_pair,
                                'model': model,
                                'fix_elites': fix_el,
                                'fix_level_seeds': fix_seed,
                                'exp_name': exp_name,
                                'save_levels': True,
                            })
                            if EVALUATE:
                                exp_config.update({
                                    'infer': True,
                                    'evaluate': True,
                                    'render_levels': True,
                                    'save_levels': True,
                                    'visualize': True,
                                })
                            print('Saving experiment config:\n{}'.format(exp_config))
                            with open('configs/evo/settings_{}.json'.format(i), 'w') as f:
                                json.dump(exp_config, f, ensure_ascii=False, indent=4)
                            # Launch the experiment. It should load the saved settings

                            if TEST:
                                os.system('python evolve.py -la {}'.format(i))
                                os.system('ray stop')
                            else:
                                os.system('sbatch evo_train.sh')
                            i += 1

if __name__ == '__main__':
    opts = argparse.ArgumentParser(
        description='Launch a batch of experiments/evaluations for evo-pcgrl')

    opts.add_argument(
        '-ex',
        '--experiment_name',
        help='A name to be shared by the batch of experiments.',
        default='test_0',
    )
    opts.add_argument(
        '-ev',
        '--evaluate',
        help='Evaluate a batch of evolution experiments.',
        action='store_true',
    )
    opts.add_argument(
        '-t',
        '--test',
        help='Test the batch script, i.e. run it on a local machine and evolve for minimal number of generations.',
        action='store_true',
    )
    opts = opts.parse_args()
    EXP_NAME = opts.experiment_name
    EVALUATE = opts.evaluate
    TEST = opts.test

    launch_batch(EXP_NAME)
