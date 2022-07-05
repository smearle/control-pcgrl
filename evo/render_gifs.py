import os
from pdb import set_trace as TT
import re

import imageio
import numpy as np
from PIL import Image

from evo.args import get_args, get_exp_dir, get_exp_name

RENDER_GRID = True

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
        n_steps = settings['n_steps']
        args, arg_dict = get_args(load_args=settings)
        exp_name = get_exp_name(args, arg_dict)
        exp_dir = get_exp_dir(exp_name)
        if not os.path.isdir(exp_dir):
            print('Skipping experiment, as directory does not exist: ', exp_dir)
            continue
        render_path = os.path.join(exp_dir, 'renders')
        if not os.path.isdir(render_path):
            continue
        model_dirs = [m for m in os.listdir(render_path) if os.path.isdir(os.path.join(render_path, m)) and 'model' in m]
        model_idxs = [re.match(r".*_(\d)_(\d)", m).groups() for m in model_dirs]
        model_idxs = [(int(m0), int(m1)) for (m0, m1) in model_idxs]
        grid_tiles_w = max([m[0] for m in model_idxs])
        grid_tiles_h = max([m[1] for m in model_idxs])
        model_frame_seqs = []
        for m_dir, m_idx in zip(model_dirs, model_idxs):
            model_path = os.path.join(render_path, m_dir)
            model_dirs = os.listdir(model_path)
            model_dirs.sort(key=natural_keys)
            frames = [os.path.join(model_path, m) for m in model_dirs if m.endswith('png') and re.match(r"frame.*.png", m)]
            if len(frames) < n_steps:
                frames += [frames[-1]] *(n_steps - len(frames))
            if RENDER_GRID:
                model_frame_seqs.append(frames)
#           os.system("ffmpeg -r 30 -i \"{0}.gif\" -movflags faststart -pix_fmt yuv420p -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" \"{0}.mp4\"".format(gif_name))
            elif len(frames) > 0:
                gif_name = os.path.join(render_path, '{}'.format(m_dir))
                frames_to_gif('{}.gif'.format(gif_name), frames)
                os.system("ffmpeg -y -r 30 -f gif -i \"{0}.gif\" -pix_fmt yuv420p \"{0}.mp4\"".format(gif_name))

            else:
                print("No gif created, no frames gathered.")
#           curdir = os.path.abspath(os.curdir)
#           os.chdir(model_path)
#           os.system("ffmpeg -y -r 30 -i frame_%04d.png \"{}.mp4\"".format(gif_name.split('/')[-1]))
#           os.chdir(curdir)
        im_w, im_h = None, None
        if RENDER_GRID:
            grid_frames = []
            grid_frames_dir = os.path.join(render_path, 'grid_frames')
            if not os.path.isdir(grid_frames_dir):
                os.mkdir(grid_frames_dir)
            for i, frames in enumerate(zip(*model_frame_seqs)):
                ims = [imageio.imread(f) for f in frames]
                if im_w is None:
                    im_w, im_h = ims[0].shape[0], ims[0].shape[1]
                grid_frame = np.empty(shape=(im_w*(grid_tiles_w+1), im_h*(grid_tiles_h+1), 3), dtype=np.uint8)
                print(grid_frame.shape)
                for j, im in enumerate(ims):
#               grid_frame = Image.new(mode="RGBA", size=(grid_w, grid_h))
                    x, y = model_idxs[j]
                    grid_frame[(-y-1)*im_w: (grid_tiles_w+1-y)*im_w, (x)*im_h: (x+1)*im_h, :] = np.array(im)
                grid_frame = Image.fromarray(grid_frame, mode="RGB")
                grid_frame.save(open(os.path.join(grid_frames_dir, 'frame_{:04d}.png'.format(i)), mode='wb'))
            gif_name = os.path.join(grid_frames_dir)
#           with imageio.get_writer(gif_name, mode='I') as writer:
#               for im in grid_frames:
#                   writer.append_data(np.array(im))
            print(gif_name)
#           frames_to_gif('{}.gif'.format(gif_name), grid_frames)
            os.system("ffmpeg -y -r 15 -i \"{0}/frame_%04d.png\" -vf tpad=stop_mode=clone:stop_duration=1 \"{0}\".gif".format(grid_frames_dir))
            os.system("ffmpeg -y -r 10 -f gif -i \"{0}.gif\" -c:v libx264 -crf 20 -pix_fmt yuv420p -vf tpad=stop_mode=clone:stop_duration=1 \"{0}.mp4\"".format(gif_name))


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
