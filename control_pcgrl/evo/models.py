import math
from pdb import set_trace as TT

import cv2
from einops import rearrange
import neat
from neat import DefaultGenome
import numpy as np
from pytorch_neat.cppn import create_cppn, Leaf
from qdpy import phenotype
import torch as th
from torch import nn
from torch.nn import Conv2d, Conv3d, Linear

from cbam import CBAM
from utils import get_one_hot_map, draw_net


class ResettableNN(nn.Module):
    """Neural networks that may have internal state that needs to be reset.
    
    For example, NCAs with "auxiliary" activations---channels in the map that are not actually part of the level, but
    used as external memory by the model (therefore, we store them here). Or maybe memory states? Same thing??"""
    def __init__(self, step_size=0.01, **kwargs):
        self.step_size = step_size
        super().__init__()

    def reset(self):
        pass

    def mutate(self):
        set_nograd(self)
        w = get_init_weights(self, init=False, torch=True)

        # Add a random gaussian to the weights, with mean 0 and standard deviation `self.step_size`.
        w += th.randn_like(w) * math.sqrt(self.step_size)

        set_weights(self, w)


class MixActiv(nn.Module):
    def __init__(self):
        super().__init__()
        self.activations = (th.sin, th.tanh, gauss, th.relu)
        self.n_activs = len(self.activations)


    def forward(self, x):
        n_chan = x.shape[1]
        chans_per_activ = n_chan / self.n_activs
        chan_i = 0
        xs = []
        for i, activ in enumerate(self.activations):
            xs.append(activ(x[:, int(chan_i):int(chan_i + chans_per_activ), :, :]))
            chan_i += chans_per_activ
        x = th.cat(xs, axis=1)
        return x


RENDER_AUX_NCA = False # overwrite kwarg so we can render level frames on HPC

class NCA(ResettableNN):
    def __init__(self, n_in_chans, n_actions, n_aux_chan=0, render=False, **kwargs):
        """
        Args:
            render (bool): whether to render the auxiliary channels in order to observe the model's behavior.
        """
        super().__init__(**kwargs)
        self._has_aux = n_aux_chan > 0
        self.n_hid_1 = n_hid_1 = 32
        self.n_aux = n_aux_chan
        self.l1 = Conv2d(n_in_chans + self.n_aux, n_hid_1, 3, 1, 1, bias=True)
        self.l2 = Conv2d(n_hid_1, n_hid_1, 1, 1, 0, bias=True)
        self.l3 = Conv2d(n_hid_1, n_actions + self.n_aux, 1, 1, 0, bias=True)
        self.layers = [self.l1, self.l2, self.l3]
        self.last_aux = None
        self.apply(init_weights)
        # self._render = render
        self._render = RENDER_AUX_NCA
        if self._render:
            cv2.namedWindow("Auxiliary NCA")

    def forward(self, x):
        with th.no_grad():
            if self._has_aux:
                if self.last_aux is None:
                    self.last_aux = th.zeros(size=(1, self.n_aux, *x.shape[-2:]))
                x_in = th.cat([x, self.last_aux], axis=1)
            else:
                x_in = x
            x = self.l1(x_in)
            x = th.relu(x)
            x = self.l2(x)
            x = th.relu(x)
            x = self.l3(x)
            x = th.sigmoid(x)

            if self._has_aux > 0:
                self.last_aux = x[:,-self.n_aux:,:,:]
                x = x[:, :-self.n_aux,:,:]

            if self._render:
#               im = self.last_aux[0].cpu().numpy().transpose(1,2,0)
                aux = self.last_aux[0].cpu().numpy()
                aux = aux / aux.max()
                im = np.expand_dims(np.vstack(aux), axis=0)
                im = im.transpose(1, 2, 0)
                cv2.imshow("Auxiliary NCA", im)
                cv2.waitKey(1)

        # axis 0 is batch
        # axis 1 is the tile-type (one-hot)
        # axis 0,1 is the x value
        # axis 0,2 is the y value

        return x, False

    def reset(self, init_aux=None):
        self.last_aux = None


class DoneAuxNCA(NCA):
    def __init__(self, n_in_chans, n_actions, n_aux=3, **kwargs):
        # Add an extra auxiliary ("done") channel after the others
        n_aux += 1
        super().__init__(n_in_chans, n_actions, n_aux_chan=n_aux, **kwargs)
        done_kernel_size = 3
        self.l_done = Conv2d(1, 1, 7, stride=999)
        self.layers += [self.l_done]

    def forward(self, x):
        with th.no_grad():
            x, done = super().forward(x)
            # retrieve local activation from done channel
            done_x = th.sigmoid(self.l_done(x[:,-1:,:,:])).flatten() - 0.5
            done = (done_x > 0).item()

        return x, done

    def reset(self, init_aux=None):
        self.last_aux = None


class AttentionNCA(ResettableNN):
    def __init__(self, n_in_chans, n_actions, n_aux_chan=3, **kwargs):
        self.n_aux = n_aux_chan
        super().__init__(**kwargs)
        n_in_chans += n_aux_chan
        h_chan = 48
        self.l1 = Conv2d(n_in_chans, h_chan, 1, 1, 0, bias=True)
        self.cbam = CBAM(h_chan, 1)
        self.l2 = Conv2d(h_chan, n_actions + n_aux_chan, 1, 1, 0, bias=True)
#       self.layers = [getattr(self.cbam, k) for k in self.cbam.state_dict().keys()]
#       self.bn = nn.BatchNorm2d(n_actions, affine=False)
        self.layers = [self.l1, self.l2, self.cbam.ChannelGate.l1, self.cbam.ChannelGate.l2, self.cbam.SpatialGate.spatial.conv]
        self.last_aux = None

    def forward(self, x):
        with th.no_grad():
            if self.last_aux is None:
                self.last_aux = th.zeros(size=(1, self.n_aux, *x.shape[-2:]))
            x_in = th.cat([x, self.last_aux], axis=1)
            x = self.l1(x_in)
            x = th.relu(x)
            x = self.cbam(x)
            x = th.relu(x)
            x = self.l2(x)
 #          x = self.bn(x)
            x = th.sigmoid(x)
            self.last_aux = x[:,-self.n_aux:,:,:]
            x = x[:, :-self.n_aux,:,:]

        return x, False

    def reset(self, init_aux=None):
        self.last_aux = None


# TODO: Let a subclass handle case where n_aux_chan > 0
class NCA3D(ResettableNN):
    """ A neural cellular automata-type NN to generate levels or wide-representation action distributions."""

    def __init__(self, n_in_chans, n_actions, n_aux_chan=0, **kwargs):
        """A 3-dimensional Neural Cellular Automata.

        Args:
            n_in_chans (_type_): _description_
            n_actions (_type_): _description_
            n_aux (int, optional): Auxiliary channels. That is, channels in the NCA's input & output, which do not
                have any effect on the map itself, but can be used as a form of external memory. Defaults to 3.
        """
        super().__init__(**kwargs)
        n_hid_1 = 32
        self.n_aux = n_aux_chan
        self.l1 = Conv3d(n_in_chans + n_aux_chan, n_hid_1, 3, 1, 1, bias=True)
        self.l2 = Conv3d(n_hid_1, n_hid_1, 1, 1, 0, bias=True)
        self.l3 = Conv3d(n_hid_1, n_actions + n_aux_chan, 1, 1, 0, bias=True)
        self.last_aux = None
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)

    def forward(self, x):
        with th.no_grad():
            if self.n_aux > 0:
                if self.last_aux is None:
                    self.last_aux = th.zeros(size=(1, self.n_aux, *x.shape[-3:]))
                x = th.cat([x, self.last_aux], axis=1)
            x = self.l1(x)
            x = th.relu(x)
            x = self.l2(x)
            x = th.relu(x)
            x = self.l3(x)
            # TODO: try softmax
            x = th.sigmoid(x)
            if self.n_aux > 0:
                self.last_aux = x[:, -self.n_aux:, ...]
                x = x[:, :-self.n_aux, ...]

        return x, False


class NCA_old(ResettableNN):
    """ A neural cellular automata-type NN to generate levels or wide-representation action distributions."""

    def __init__(self, n_in_chans, n_actions, **kwargs):
        super().__init__(**kwargs)
        n_hid_1 = 32
        self.l1 = Conv2d(n_in_chans, n_hid_1, 3, 1, 1, bias=True)
        self.l2 = Conv2d(n_hid_1, n_hid_1, 1, 1, 0, bias=True)
        self.l3 = Conv2d(n_hid_1, n_actions, 1, 1, 0, bias=True)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)

    def forward(self, x):
        with th.no_grad():
            x = self.l1(x)
            x = th.relu(x)
            x = self.l2(x)
            x = th.relu(x)
            x = self.l3(x)
            # TODO: try softmax
            x = th.sigmoid(x)


        # axis 0 is batch
        # axis 1 is the tile-type (one-hot)
        # axis 0,1 is the x value
        # axis 0,2 is the y value

        return x, False


class Decoder(ResettableNN):
    """
    Decoder-like architecture (e.g. as in VAEs and GANs).
    """
    def __init__(self, n_in_chans, n_actions, n_latents=2, **kwargs):
        super().__init__(**kwargs)
        n_hid_1 = 16
        self.l1 = nn.ConvTranspose2d(n_in_chans + n_latents, n_hid_1, 3, 2, 1, 1, bias=True)
        self.l2 = nn.ConvTranspose2d(n_hid_1, n_hid_1, 3, 2, 1, 1, bias=True)
        self.l3 = Conv2d(n_hid_1, n_actions, 1, 1, 0, bias=True)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)

    def forward(self, x):
        with th.no_grad():
            # Observe the coordinates
            coords = get_coord_grid(x, normalize=True)
            x = th.hstack((coords, x))
            x = self.l1(x)
            x = th.relu(x)
            x = self.l2(x)
            x = th.relu(x)
            x = self.l3(x)
            # TODO: try softmax
            x = th.sigmoid(x)

        return x, False


class DeepDecoder(ResettableNN):
    """
    Decoder-like architecture (e.g. as in VAEs and GANs). But deeper.
    """
    def __init__(self, n_in_chans, n_actions, **kwargs):
        super().__init__(**kwargs)
        n_hid_1 = 10
        self.l1 = nn.ConvTranspose2d(n_in_chans + 2, n_hid_1, 3, 1, 0, 0, bias=True)
        self.l2 = nn.ConvTranspose2d(n_hid_1, n_hid_1, 3, 1, 0, 0, bias=True)
        self.l3 = nn.ConvTranspose2d(n_hid_1, n_hid_1, 3, 1, 0, 0, bias=True)
        self.l4 = nn.ConvTranspose2d(n_hid_1, n_hid_1, 3, 1, 0, 0, bias=True)
        self.l5 = nn.ConvTranspose2d(n_hid_1, n_hid_1, 3, 1, 0, 0, bias=True)
        self.l6 = nn.ConvTranspose2d(n_hid_1, n_hid_1, 3, 1, 0, 0, bias=True)
        self.l7 = Conv2d(n_hid_1, n_actions, 1, 1, 0, bias=True)
        self.layers = [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7]
        self.apply(init_weights)

    def forward(self, x):
        with th.no_grad():
            # Observe the coordinates
            coords = get_coord_grid(x, normalize=True)
            x = th.hstack((coords, x))
            x = self.l1(x)
            x = th.relu(x)
            x = self.l2(x)
            x = th.relu(x)
            x = self.l3(x)
            x = th.relu(x)
            x = self.l4(x)
            x = th.relu(x)
            x = self.l5(x)
            x = th.relu(x)
            x = self.l6(x)
            x = th.relu(x)
            x = self.l7(x)
            x = th.sigmoid(x)

        return x, False


class MixNCA(ResettableNN):
    def __init__(self, *args, **kwargs):
        super(MixNCA, self).__init__(**kwargs)
        self.mix_activ = MixActiv()

    def forward(self, x):
        with th.no_grad():
            x = self.l1(x)
            x = self.mix_activ(x)
            x = self.l2(x)
            x = self.mix_activ(x)
            x = self.l3(x)
            x = th.sigmoid(x)


class CoordNCA(ResettableNN):
    """ A neural cellular automata-type NN to generate levels or wide-representation action distributions.
    With coordinates as additional input, like a CPPN."""

    def __init__(self, n_in_chans, n_actions, **kwargs):
        super().__init__(**kwargs)
        n_hid_1 = 28
        #       n_hid_2 = 16

        self.l1 = Conv2d(n_in_chans + 2, n_hid_1, 3, 1, 1, bias=True)
        self.l2 = Conv2d(n_hid_1, n_hid_1, 1, 1, 0, bias=True)
        self.l3 = Conv2d(n_hid_1, n_actions, 1, 1, 0, bias=True)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)

    def forward(self, x):
        with th.no_grad():
            coords = get_coord_grid(x, normalize=True)
            x = th.hstack((coords, x))
            x = self.l1(x)
            x = th.relu(x)
            x = self.l2(x)
            x = th.relu(x)
            x = self.l3(x)
            x = th.sigmoid(x)

        # axis 0 is batch
        # axis 1 is the tile-type (one-hot)
        # axis 0,1 is the x value
        # axis 0,2 is the y value

        return x, False


def get_coord_grid(x, normalize=False, env3d=False):
    if env3d:
        length = x.shape[-3]
    width = x.shape[-2]
    height = x.shape[-1]

    X = th.arange(width)
    Y = th.arange(height)

    if env3d:
        Z = th.arange(length)

    if normalize:
        X = X / width
        Y = Y / height
        if env3d:
            Z = Z / length
    else:
        X = X / 1
        Y = Y / 1
        if env3d:
            Z = Z / 1
    if not env3d:
        X, Y = th.meshgrid(X, Y)
        x = th.stack((X, Y)).unsqueeze(0)
    else:
        X, Y, Z = th.meshgrid(X, Y, Z)
        x = th.stack((X, Y, Z)).unsqueeze(0)

    return x


class FeedForwardCPPN(nn.Module):
    def __init__(self, n_in_chans, n_actions, **kwargs):
        super().__init__(**kwargs)
        n_hid = 64
        self.l1 = Conv2d(2, n_hid, kernel_size=1)
        self.l2 = Conv2d(n_hid, n_hid, kernel_size=1)
        self.l3 = Conv2d(n_hid, n_actions, kernel_size=1)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)

    def forward(self, x):
        x = get_coord_grid(x, normalize=True)
        with th.no_grad():
            x = th.relu(self.l1(x))
            x = th.relu(self.l2(x))
            x = th.sigmoid(self.l3(x))

        return x, True


class GenReluCPPN(ResettableNN):
    def __init__(self, n_in_chans, n_actions, **kwargs):
        super().__init__(**kwargs)
        n_hid = 64
        self.l1 = Conv2d(2+n_in_chans, n_hid, kernel_size=1)
        self.l2 = Conv2d(n_hid, n_hid, kernel_size=1)
        self.l3 = Conv2d(n_hid, n_actions, kernel_size=1)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)

    def forward(self, x):
        coord_x = get_coord_grid(x, normalize=True)
        x = th.cat((x, coord_x), axis=1)
        with th.no_grad():
            x = th.relu(self.l1(x))
            x = th.relu(self.l2(x))
            x = th.sigmoid(self.l3(x))

        return x, True


class SinCPPN(ResettableNN):
    """A vanilla CPPN that only takes (x, y) coordinates. #TODO: merge with GenSinCPPN"""
    def __init__(self, n_in_chans, n_actions, n_latents=2, **kwargs):
        super().__init__(**kwargs)
        n_hid = 64
        self.l1 = Conv2d(n_latents, n_hid, kernel_size=1)
        self.l2 = Conv2d(n_hid, n_hid, kernel_size=1)
        self.l3 = Conv2d(n_hid, n_actions, kernel_size=1)
        self.layers = [self.l1, self.l2, self.l3]
        # if "Sin2" in MODEL:
        # print('init_siren')
        init_siren_weights(self.layers[0], first_layer=True)
        [init_siren_weights(li, first_layer=False) for li in self.layers[1:]]
        # else:
            # self.apply(init_weights)

    def forward(self, x):
        x = get_coord_grid(x, normalize=True) * 2
        with th.no_grad():
            x = th.sin(self.l1(x))
            x = th.sin(self.l2(x))
            x = th.sigmoid(self.l3(x))

        return x, True


class GenSinCPPN(ResettableNN):
    def __init__(self, n_in_chans, n_actions, n_latents=2, **kwargs):
        super().__init__(**kwargs)
        n_hid = 64
        self.l1 = Conv2d(n_latents+n_in_chans, n_hid, kernel_size=1)
        self.l2 = Conv2d(n_hid, n_hid, kernel_size=1)
        self.l3 = Conv2d(n_hid, n_actions, kernel_size=1)
        self.layers = [self.l1, self.l2, self.l3]
        # if "Sin2" in MODEL:
        init_siren_weights(self.layers[0], first_layer=True)
        [init_siren_weights(li, first_layer=False) for li in self.layers[1:]]
        # else:
            # self.apply(init_weights)

    def forward(self, x):
        coord_x = get_coord_grid(x, normalize=True) * 2
        x = th.cat((x, coord_x), axis=1)
        with th.no_grad():
            x = th.sin(self.l1(x))
            x = th.sin(self.l2(x))
            x = th.sigmoid(self.l3(x))

        return x, True


class MixCPPN(ResettableNN):
    def __init__(self, n_in_chans, n_actions, **kwargs):
        super().__init__(**kwargs)
        n_hid = 64
        self.l1 = Conv2d(2, n_hid, kernel_size=1)
        self.l2 = Conv2d(n_hid, n_hid, kernel_size=1)
        self.l3 = Conv2d(n_hid, n_actions, kernel_size=1)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)
        self.mix_activ = MixActiv()


    def forward(self, x):
        x = get_coord_grid(x, normalize=True) * 2
        with th.no_grad():
            x = self.mix_activ(self.l1(x))
            x = self.mix_activ(self.l2(x))
            x = th.sigmoid(self.l3(x))

        return x, True


class GenMixCPPN(ResettableNN):
    def __init__(self, n_in_chans, n_actions, **kwargs):
        super().__init__(**kwargs)
        n_hid = 64
        self.l1 = Conv2d(2+n_in_chans, n_hid, kernel_size=1)
        self.l2 = Conv2d(n_hid, n_hid, kernel_size=1)
        self.l3 = Conv2d(n_hid, n_actions, kernel_size=1)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)
        self.mix_activ = MixActiv()


    def forward(self, x):
        coord_x = get_coord_grid(x, normalize=True) * 2
        x = th.cat((x, coord_x), axis=1)
        with th.no_grad():
            x = self.mix_activ(self.l1(x))
            x = self.mix_activ(self.l2(x))
            x = th.sigmoid(self.l3(x))

        return x, True


class FixedGenCPPN(ResettableNN):
    """A fixed-topology CPPN that takes additional channels of noisey input to prompts its output.
    Like a CoordNCA but without the repeated passes and with 1x1 rather than 3x3 kernels."""
    # TODO: Maybe try this with 3x3 conv, just to cover our bases?
    def __init__(self, n_in_chans, n_actions, **kwargs):
        super().__init__(**kwargs)
        n_hid = 64
        self.l1 = Conv2d(2 + n_in_chans, n_hid, kernel_size=1)
        self.l2 = Conv2d(n_hid, n_hid, kernel_size=1)
        self.l3 = Conv2d(n_hid, n_actions, kernel_size=1)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)

    def forward(self, x):
        coord_x = get_coord_grid(x, normalize=True) * 2
        x = th.cat((x, coord_x), axis=1)
        with th.no_grad():
            x = th.sin(self.l1(x))
            x = th.sin(self.l2(x))
            x = th.sigmoid(self.l3(x))

        return x, True


class DirectEncoding():
    """In this "model" (if you can call it that), the weights are the level itself. (Continuous methods like CMA won't
    make sense here!) Though we accept an input, it is totally ignored."""
    def __init__(self, n_in_chans, n_actions, map_dims, **kwargs):
        self.n_actions = n_actions  # how many distinct tiles can appear in the output
        self.layers = np.array([])  # dummy
        self.discrete = th.randint(0, n_in_chans, map_dims[::-1])

    def __call__(self, x):
        # onehot = th.zeros(1, x.shape[1], x.shape[2], x.shape[3])
        # onehot[0,0,self.discrete==0]=1
        # onehot[0,1,self.discrete==1]=1
        onehot = th.eye(self.n_actions)[self.discrete]
        onehot = onehot.unsqueeze(0)
        onehot = onehot.transpose(1, -1)
        # onehot = rearrange(onehot, "h w c -> 1 c h w")  # prettier, but not dimension agnostic (?)
        return onehot, True

    def mutate(self):
        # flip some tiles
        mut_act = (th.rand(self.discrete.shape) < 0.01).long()  # binary mutation actions
        mut_act *= th.randint(1, self.n_actions, mut_act.shape)
        new_discrete = (self.discrete + mut_act) % self.n_actions
        self.discrete = new_discrete

    def reset(self):
        return


neat_config_path = 'evo/config_cppn'


class CPPN(ResettableNN):
    def __init__(self, n_in_chans, n_actions, **kwargs):
        super().__init__(**kwargs)
        self.neat_config = neat.config.Config(DefaultGenome, neat.reproduction.DefaultReproduction,
                                              neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                                              neat_config_path)
        self.n_actions = n_actions
        self.neat_config.genome_config.num_outputs = n_actions
        self.neat_config.genome_config.num_hidden = 2
        self.genome = DefaultGenome(0)
        self.genome.configure_new(self.neat_config.genome_config)
        self.input_names = ['x_in', 'y_in']
        self.output_names = ['tile_{}'.format(i) for i in range(n_actions)]
        self.cppn = create_cppn(self.genome, self.neat_config, self.input_names, self.output_names)

    def mate(self, ind_1, fit_0, fit_1):
        self.genome.fitness = fit_0
        ind_1.genome.fitness = fit_1
        return self.genome.configure_crossover(self.genome, ind_1.genome, self.neat_config.genome_config)

    def mutate(self):
#       print(self.input_names, self.neat_config.genome_config.input_keys, self.genome.nodes)
        self.genome.mutate(self.neat_config.genome_config)
        self.cppn = create_cppn(self.genome, self.neat_config, self.input_names, self.output_names)

    def draw_net(self):
        draw_net(self.neat_config, self.genome,  view=True, filename='cppn')

    def forward(self, x):
        X = th.arange(x.shape[-2])
        Y = th.arange(x.shape[-1])
        X, Y = th.meshgrid(X/X.max(), Y/Y.max())
        tile_probs = [self.cppn[i](x_in=X, y_in=Y) for i in range(self.n_actions)]
        multi_hot = th.stack(tile_probs, axis=0)
        multi_hot = multi_hot.unsqueeze(0)
        return multi_hot, True


class CPPNCA(ResettableNN):
    def __init__(self, n_in_chans, n_actions, **kwargs):
        super().__init__(**kwargs)
        n_hid_1 = 32
        with th.no_grad():
            self.l1 = Conv2d(n_in_chans, n_hid_1, 3, 1, 1, bias=True)
            self.l2 = Conv2d(n_hid_1, n_hid_1, 1, 1, 0, bias=True)
            self.l3 = Conv2d(n_hid_1, n_actions, 1, 1, 0, bias=True)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)
        n_nca_params = sum(p.numel() for p in self.parameters())
        self.cppn_body = GenCPPN(n_in_chans, n_actions)
        self.normal = th.distributions.multivariate_normal.MultivariateNormal(th.zeros(1), th.eye(1))

    def mate(self):
        raise NotImplementedError

    def mutate(self):
        self.cppn_body.mutate()

        with th.no_grad():
            for layer in self.layers:
                dw = self.normal.sample(layer.weight.shape)
                layer.weight = th.nn.Parameter(layer.weight + dw.squeeze(-1))
                db = self.normal.sample(layer.bias.shape)
                layer.bias = th.nn.Parameter(layer.bias + db.squeeze(-1))

    def forward(self, x):
        with th.no_grad():
            x = self.l1(x)
            x = th.relu(x)
            x = self.l2(x)
            x = th.relu(x)
            x = th.sigmoid(x)
        x, _ = self.cppn_body(x)
        return x, False


class GenCPPN(CPPN):
    def __init__(self, n_in_chans, n_actions, **kwargs):
        super().__init__(n_in_chans, n_actions, **kwargs)
        self.neat_config = neat.config.Config(DefaultGenome, neat.reproduction.DefaultReproduction,
                                              neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                                              neat_config_path)
        self.n_actions = n_actions
        self.neat_config.genome_config.num_outputs = n_actions
        self.genome = DefaultGenome(0)
        self.input_names = ['x_in', 'y_in'] + ['tile_{}_in'.format(i) for i in range(n_actions)]
        n_inputs = len(self.input_names)
        self.output_names = ['tile_{}_out'.format(i) for i in range(n_actions)]
        self.neat_config.genome_config.input_keys = (-1*np.arange(n_inputs) - 1).tolist()
        self.neat_config.genome_config.num_inputs = n_inputs
        self.neat_config.genome_config.num_hidden = 2
        self.genome.configure_new(self.neat_config.genome_config)
        self.cppn = create_cppn(self.genome, self.neat_config, self.input_names, self.output_names)

    def forward(self, x):
        x[:, :, :, :] = x[:, :, 0:1, 0:1]
        X = th.arange(x.shape[-2])
        Y = th.arange(x.shape[-1])
        X, Y = th.meshgrid(X/X.max(), Y/Y.max())
        inputs = {'x_in': X, 'y_in': Y}
        inputs.update({'tile_{}_in'.format(i): th.Tensor(x[0,i,:,:]) for i in range(self.n_actions)})
        tile_probs = [self.cppn[i](**inputs) for i in range(self.n_actions)]
        multi_hot = th.stack(tile_probs, axis=0)
        multi_hot = multi_hot.unsqueeze(0)
        multi_hot = (multi_hot + 1) / 2
        return multi_hot, True


Sin2CPPN = SinCPPN
GenSinCPPN2 = GenSinCPPN
GenSin2CPPN2 = GenSinCPPN
GenCPPN2 = GenCPPN


def gauss(x, mean=0, std=1):
    return th.exp((-(x - mean) ** 2)/(2 * std ** 2))


class Individual(phenotype.Individual):
    "An individual for mutating with operators. Assuming we're using vanilla MAP-Elites here."
    def __init__(self, model_cls, n_in_chans, n_actions, **kwargs):
        super(Individual, self).__init__()
        self.model = model_cls(n_in_chans, n_actions, **kwargs)
        # Provide weight of 1 to prevent qdpy/deap from defaulting to minimization
        self.fitness = phenotype.Fitness([0], weights=[1])
        self.fitness.delValues()

    def mutate(self):
        self.model.mutate()

    def mate(self, ind_1):
        assert len(self.fitness.values) == 1 == len(ind_1.fitness.values)
        self.model.mate(ind_1.model, fit_0=self.fitness.values[0], fit_1=ind_1.fitness.values[0])

    def __eq__(self, ind_1):
        if not hasattr(ind_1, "model"): return False
        return self.model == ind_1.model


class GeneratorNNDenseSqueeze(ResettableNN):
    """ A neural cellular automata-type NN to generate levels or wide-representation action distributions."""

    def __init__(self, n_in_chans, n_actions, observation_shape, n_flat_actions, **kwargs):
        super().__init__(**kwargs)
        n_hid_1 = 16
        # Hack af. Pad the input to make it have root 2? idk, bad
        sq_i = 2
        assert observation_shape[-1] == observation_shape[-2]

        #       while sq_i < observation_shape[-1]:
        #           sq_i = sq_i**2
        #       pad_0 = sq_i - observation_shape[-1]
        self.l1 = Conv2d(n_in_chans, n_hid_1, 3, 1, 0, bias=True)
        self.l2 = Conv2d(n_hid_1, n_hid_1, 3, 2, 0, bias=True)
        self.flatten = th.nn.Flatten()
        n_flat = self.flatten(
            self.l2(self.l1(th.zeros(size=observation_shape)))
        ).shape[-1]
        # n_flat = n_hid_1
        self.d1 = Linear(n_flat, n_flat_actions)
        #       self.d2 = Linear(16, n_flat_actions)
        self.layers = [self.l1, self.l2, self.d1]
        self.apply(init_weights)

    def forward(self, x):
        with th.no_grad():
            x = self.l1(x)
            x = th.relu(x)
            x = self.l2(x)
            x = th.relu(x)
            #           for i in range(int(np.log2(x.shape[2])) + 1):
            #               x = self.l2(x)
            #               x = th.nn.functional.relu(x)
            x = self.flatten(x)
            x = self.d1(x)
            x = th.sigmoid(x)
            #           x = self.d2(x)
            #           x = th.sigmoid(x)

        return x, False


class GeneratorNNDense(ResettableNN):
    """ A neural cellular automata-type NN to generate levels or wide-representation action distributions."""

    def __init__(self, n_in_chans, n_actions, observation_shape, n_flat_actions, **kwargs):
        super().__init__(**kwargs)
        n_hid_1 = 16
        n_hid_2 = 32
        self.conv1 = Conv2d(n_in_chans, n_hid_1, kernel_size=3, stride=2)
        self.conv2 = Conv2d(n_hid_1, n_hid_2, kernel_size=3, stride=2)
        self.conv3 = Conv2d(n_hid_2, n_hid_2, kernel_size=3, stride=2)
        self.flatten = th.nn.Flatten()
        n_flat = self.flatten(
            self.conv3(self.conv2(self.conv1(th.zeros(size=observation_shape))))
        ).shape[-1]
        #       self.fc1 = Linear(n_flat, n_flat_actions)
        self.fc1 = Linear(n_flat, n_hid_2)
        self.fc2 = Linear(n_hid_2, n_flat_actions)
        self.layers = [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]
        self.apply(init_weights)

    def forward(self, x):
        with th.no_grad():
            x = th.relu(self.conv1(x))
            x = th.relu(self.conv2(x))
            x = th.relu(self.conv3(x))
            x = self.flatten(x)
            x = th.relu(self.fc1(x))
            x = th.softmax(self.fc2(x), dim=1)

        return x, False


class PlayerNN(ResettableNN):
    def __init__(self, n_tile_types, n_actions=4, **kwargs):
        super().__init__(**kwargs)
        self.n_tile_types = n_tile_types
        # assert "zelda" in PROBLEM
        self.l1 = Conv2d(n_tile_types, 16, 3, 1, 0, bias=True)
        self.l2 = Conv2d(16, 16, 3, 2, 1, bias=True)
        self.l3 = Conv2d(16, n_actions, 3, 1, 1, bias=True)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_play_weights)
        self.flatten = th.nn.Flatten()
        self.net_reward = 0
        self.n_episodes = 0

    def forward(self, x):
        x = th.Tensor(get_one_hot_map(x, self.n_tile_types))
        x = x.unsqueeze(0)
        with th.no_grad():
            x = th.relu(self.l1(x))

            for i in range(int(np.log2(x.shape[2])) + 1):
                #           for i in range(1):
                x = th.relu(self.l2(x))
            x = th.relu(self.l3(x))

            #           x = x.argmax(1)
            #           x = x[0]
            x = x.flatten()
            x = th.softmax(x, axis=0)
            # x = [x.argmax().item()]
            act_ids = np.arange(x.shape[0])
            probs = x.detach().numpy()
            x = np.random.choice(act_ids, 1, p=probs)

        return x

    def assign_reward(self, rew):
        self.net_reward += rew
        self.n_episodes += 1

    def reset(self):
        self.net_reward = 0
        self.n_episodes = 0

    def get_reward(self):
        mean_rew = self.net_reward / self.n_episodes

        return mean_rew


def init_siren_weights(m, first_layer=False):
    if first_layer:
        th.nn.init.constant_(m.weight, 30)
        return
    if type(m) == th.nn.Conv2d:
        ws = m.weight.shape
        # number of _inputs
        n = ws[0] * ws[1] * ws[2]
        th.nn.init.uniform_(m.weight, -np.sqrt(6/n), np.sqrt(6/n))
        m.bias.data.fill_(0.01)
    else:
        raise Exception


def init_weights(m):
    if type(m) == th.nn.Linear:
        th.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

    if type(m) == th.nn.Conv2d:
        th.nn.init.orthogonal_(m.weight)


def init_play_weights(m):
    if type(m) == th.nn.Linear:
        th.nn.init.xavier_uniform(m.weight, gain=0)
        m.bias.data.fill_(0.00)

    if type(m) == th.nn.Conv2d:
        #       th.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        th.nn.init.constant_(m.weight, 0)


def set_nograd(nn):
    if not hasattr(nn, "parameters"):
        return
    for param in nn.parameters():
        param.requires_grad = False


def get_init_weights(nn, init=True, torch=False):
    """
    Use to get flat vector of weights from PyTorch model
    """
    init_params = []

    if isinstance(nn, CPPN):
        for node in nn.cppn:
            if isinstance(node, Leaf):
                continue
            init_params.append(node.weights)
            init_params.append(node.bias)
    else:
        for lyr in nn.layers:
            init_params.append(lyr.weight.view(-1))
            if lyr.bias is not None:
                init_params.append(lyr.bias.view(-1))
    if not torch:
        init_params = [p.cpu().numpy() for p in init_params]
        init_params = np.hstack(init_params)
    else:
        init_params = th.cat(init_params)
    if init:
        print("number of initial NN parameters: {}".format(init_params.shape))

    return init_params


def set_weights(nn, weights, algo="CMAME"):
    if algo == "ME":
        # then out nn is contained in the individual
        individual = weights  # I'm sorry mama
        return individual.model
    with th.no_grad():
        n_el = 0

        if isinstance(nn, CPPN):
            for node in nn.cppn:
                l_weights = weights[n_el : n_el + len(node.weights)]
                n_el += len(node.weights)
                node.weights = l_weights
                b_weight = weights[n_el: n_el + 1]
                n_el += 1
                node.bias = b_weight
        else:
            for layer in nn.layers:
                l_weights = weights[n_el : n_el + layer.weight.numel()]
                n_el += layer.weight.numel()
                l_weights = l_weights.reshape(layer.weight.shape)
                layer.weight = th.nn.Parameter(th.Tensor(l_weights))
                layer.weight.requires_grad = False
                if layer.bias is not None:
                    n_bias = layer.bias.numel()
                    b_weights = weights[n_el : n_el + n_bias]
                    n_el += n_bias
                    b_weights = b_weights.reshape(layer.bias.shape)
                    layer.bias = th.nn.Parameter(th.Tensor(b_weights))
                    layer.bias.requires_grad = False

    return nn