import os
from pdb import set_trace as TT
import re

import imageio

from evo_args import get_args, get_exp_name

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def render_gifs(settings_list):
    batch_exp_name = settings_list[0]["exp_name"]
    for i, settings in enumerate(settings_list):
        args, arg_dict = get_args(load_args=settings)
        exp_name = get_exp_name(args, arg_dict)
        if not os.path.isdir(exp_name):
            print('Skipping experiment, as directory does not exist: ', exp_name)
            continue
        render_path = os.path.join(exp_name, 'renders')
        if not os.path.isdir(render_path):
            continue
        model_dirs = [m for m in os.listdir(render_path) if os.path.isdir(os.path.join(render_path, m)) and 'model' in m]
        for m_dir in model_dirs:
            model_path = os.path.join(render_path, m_dir)
            model_dirs = os.listdir(model_path)
            model_dirs.sort(key=natural_keys)
            frames = [os.path.join(model_path, m) for m in model_dirs if m.endswith('png') and re.match(r"frame.*.png", m)]
            gif_name = os.path.join(render_path, '{}'.format(m_dir))
            frames_to_gif('{}.gif'.format(gif_name), frames)
#           os.system("ffmpeg -r 30 -i \"{0}.gif\" -movflags faststart -pix_fmt yuv420p -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" \"{0}.mp4\"".format(gif_name))
            os.system("ffmpeg -y -r 30 -f gif -i \"{0}.gif\" \"{0}.mp4\"".format(gif_name))
#           os.system("ffmpeg -y -r 30 -i \"{0}.gif\" \"{0}/frame_.png\"".format(os.path.join(model_path, m)))

def frames_to_gif(gif_path, filenames):
    # Repeat the last frame a bunch, so that we "pause" on the final generated level
    filenames = filenames + [filenames[-1]] * 20
    with imageio.get_writer(gif_path, mode='I') as writer:
        for filename in filenames:
            try:
                image = imageio.imread(filename)
            except ValueError:
                print('Failed to read image {}, aborting.'.format(filename))
                return
            writer.append_data(image)
    print(gif_path)
