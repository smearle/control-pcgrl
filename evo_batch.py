'''
Launch a batch of experiments on a SLURM cluster.
'''
import os
import copy
import json

with open('configs/evo/default_settings.json', 'r') as f:
    default_config = json.load(f)

print('Loaded default config:\n{}'.format(default_config))

problems = [
    'binary_ctrl',
    'zelda_ctrl',
    'sokoban_ctrl',
    'smb_ctrl',
    ]
representations = [
    'cellular',
    'wide',
    'narrow',
    'turtle',
    ]
global_bcs = [
    ['NONE'],
    ['emptiness', 'symmetry'],
]
local_bcs = {
    'binary_ctrl': [
        ['regions', 'path-length'],
    ],
    'zelda_ctrl': [
        ['neareset-enemy', 'path-length'],
    ],
    'sokoban_ctrl': [
        ['crate', 'sol-length'],
    ],
    'smb_ctrl': [
        ['enemies', 'jumps'],
    ],
}

def launch_batch(exp_name):
    for prob in problems:
        prob_bcs = global_bcs + local_bcs[prob]
        for rep in representations:
            for bc_pair in prob_bcs:
                exp_config = copy.deepcopy(default_config)
                exp_config.update({
                    'problem': prob,
                    'representation': rep,
                    'behavior_characteristics': bc_pair,
                    'e': exp_name,
                })
                print('Saving experiment config:\n{}'.format(exp_config))
                with open('configs/evo/settings.json', 'w') as f:
                    json.dump(exp_config, f, ensure_ascii=False, indent=4)
                # Launch the experiment. It should load the saved settings
                os.system('sbatch evo_train.sh')


exp_name = '0'

if __name__ == '__main__':
    launch_batch(exp_name)
