'''
Launch a batch of experiments on a SLURM cluster.
'''
import os
import copy
import json
import re



problems = [
#   'binary_ctrl',
    'zelda_ctrl',
#   'sokoban_ctrl',
#   'smb_ctrl',
    ]
representations = [
    'cellular',
    'wide',
    'narrow',
    'turtle',
    ]
global_bcs = [
    ['NONE'],
#   ['emptiness', 'symmetry'],
]
local_bcs = {
    'binary_ctrl': [
        ['regions', 'path-length'],
    ],
    'zelda_ctrl': [
        ['nearest-enemy', 'path-length'],
    ],
    'sokoban_ctrl': [
        ['crate', 'sol-length'],
    ],
    'smb_ctrl': [
        ['enemies', 'jumps'],
    ],
}
models = [
    'NCA',
    'CNN',
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
        default_config['-ng'] = 1
    i = 0
    for prob in problems:
        prob_bcs = global_bcs + local_bcs[prob]
        for rep in representations:
            for bc_pair in prob_bcs:
                for model in models:
                    # This would necessitate an explosive number of model params so we'll not run it
                    if model == 'CNN' and rep == 'cellular':
                        continue

                    with open('evo_train.sh', 'r') as f:
                        content = f.read()
                        new_content = re.sub(
                            'python evolve.py -la \d+',
                            'python evolve.py -la {}'.format(i), content)
                    with open('evo_train.sh', 'w') as f:
                        f.write(new_content)
                    exp_config = copy.deepcopy(default_config)
                    exp_config.update({
                        'problem': prob,
                        'representation': rep,
                        'behavior_characteristics': bc_pair,
                        'model': model,
                        'e': exp_name,
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

TEST = False
EXP_NAME = '0'

if __name__ == '__main__':
    launch_batch(EXP_NAME)
