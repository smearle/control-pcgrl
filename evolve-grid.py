import json

import gym
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import gym_pcgrl

# Name of problem here
PROBLEM = 'mini'
# File path to CSV file here
CSV_PATH = 'evo_runs/EvoPCGRL_mini_10-batch_10-step_0test_levels.csv'

# Env name here
env_name = '{}-wide-v0'.format(PROBLEM)
env = gym.make(env_name)
# level_json = {'level': final_levels.tolist(),'batch_reward':[batch_reward] * len(final_levels.tolist()), 'variance': [variance_penalty] * len(final_levels.tolist()), 'diversity':[diversity_bonus] * len(final_levels.tolist()),'targets':trg.tolist(), **bc_dict}
df = pd.read_csv(CSV_PATH, header=None).rename(index=str, columns={0:'level',1:'batch_reward', 2:'variance', 3:'diversity', 4:'targets'})
for i in range(5, len(df.columns)):
    df = df.rename(index=str, columns={i:'bc{}'.format(i-5)})
df = df[df['targets']==0]  # select only the valid levels 

# Render Grid

d = 6  # dimension of rows and columns
figw, figh = 16.0, 16.0
fig, axs = plt.subplots(ncols=d, nrows=d, figsize=(figw, figh))

df_g = df.sort_values(by=['bc0', 'bc1'], ascending=False)

df_g['row'] = np.floor(np.linspace(0, d, len(df_g), endpoint=False)).astype(int)

for row_num in range(d):
    row = df_g[df_g['row']==row_num]
    row = row.sort_values(by=['bc1'], ascending=True)
    row['col'] = np.arange(0,len(row), dtype=int)
    idx = np.floor(np.linspace(0,len(row)-1,d)).astype(int)
    row = row[row['col'].isin(idx)]
    row = row.drop(['row','col'], axis=1)
    # grid_models = np.array(row.loc[:,'solution_0':])
    grid_models = row['level'].tolist()
    for col_num in range(len(row)):
        level = np.zeros((5,5), dtype=int)
        for i, l_rows in enumerate(grid_models[col_num].split('], [')):
            for j, l_col in enumerate(l_rows.split(',')):
                level[i,j] = int(l_col.replace('[','').replace(']','').replace(' ',''))

        # Set map
        env._rep._map = level
        img = env.render(mode='rgb_array')
        axs[row_num,col_num].imshow(img, aspect='auto')
        axs[row_num,col_num].set_axis_off()

fig.subplots_adjust(hspace=0.01, wspace=0.01)
fig.savefig('evo_runs/test_grid.png', dpi=300)
