import os
from pdb import set_trace as TT
import re

import imageio

from evo_args import get_args, get_exp_name

def render_gifs(settings_list):
    batch_exp_name = settings_list[0]["exp_name"]
    for i, settings in enumerate(settings_list):
        args, arg_dict = get_args(load_args=settings)
        exp_name = get_exp_name(args, arg_dict)
        if not os.path.isdir(exp_name):
            print('Skipping experiment, as directory does not exist: ', exp_name)
            continue
        render_path = os.path.join(exp_name, 'renders')
        model_dirs = [m for m in os.listdir(render_path) if os.path.isdir(os.path.join(render_path, m)) and 'model' in m]
        for m_dir in model_dirs:
            model_path = os.path.join(render_path, m_dir)
            frames = [os.path.join(model_path, m) for m in sorted(os.listdir(model_path)) if m.endswith('png') and re.match(r"frame.*.png", m)]
            frames_to_gif(os.path.join(model_path, 'generate.gif'), frames)

def frames_to_gif(gif_path, filenames):
    with imageio.get_writer(gif_path, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
