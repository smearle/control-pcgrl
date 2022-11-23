from typing import Dict, List

from einops import rearrange
import numpy as np
import torch as th
from pdb import set_trace as TT
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from ray.rllib.utils.typing import ModelConfigDict, ModelWeights
from torch import nn
from torch.nn import Conv2d, Conv3d, Linear


class CustomFeedForwardModel(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 conv_filters=64,
                 fc_size=64,
                 **kwargs
                 ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        # self.obs_size = get_preprocessor(obs_space)(obs_space).size
        obs_shape = obs_space.shape
        obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
        self.fc_size = fc_size

        self.conv_1 = nn.Conv2d(obs_space.shape[-1], out_channels=conv_filters, kernel_size=7, stride=2, padding=3)
        self.conv_2 = nn.Conv2d(conv_filters, out_channels=conv_filters, kernel_size=7, stride=2, padding=3)

        self.pre_fc_size = self.conv_2(self.conv_1(th.zeros(1, *obs_shape))).reshape(1, -1).shape[-1]

        self.fc_1 = SlimFC(self.pre_fc_size, 256)
        self.fc_2 = SlimFC(256, 64)
        self.action_branch = SlimFC(self.fc_size, num_outputs)
        self.value_branch = SlimFC(self.fc_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return th.reshape(self.value_branch(self._features), [-1])

    def forward(self, input_dict, state, seq_lens):
        input = input_dict["obs"].permute(0, 3, 1, 2)  # Because rllib order tensors the tensorflow way (channel last)
        x = nn.functional.relu(self.conv_1(input.float()))
        x = nn.functional.relu(self.conv_2(x))
        x = x.reshape(x.size(0), -1)
        x = nn.functional.relu(self.fc_1(x))
        x = nn.functional.relu(self.fc_2(x))
        self._features = x
        action_out = self.action_branch(self._features)

        return action_out, []


class SeqNCA3D(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 conv_filters=64,
                 fc_size=64,
                #  n_aux_chan=0,
                 ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        self.fc_size = fc_size
        self.conv_filters = conv_filters
        obs_shape = obs_space.shape
        self.pre_fc_size = (obs_shape[-2] - 2) * (obs_shape[-3] - 2) * (obs_shape[-4] - 2) * conv_filters
        self.conv_1 = nn.Conv3d(obs_shape[-1], out_channels=conv_filters, kernel_size=3, stride=1, padding=0)
        self.fc_1 = SlimFC(self.pre_fc_size, self.fc_size)
        self.action_branch = nn.Sequential(
            SlimFC(3 * 3 * 3 * (conv_filters), self.fc_size),
            nn.ReLU(),
            SlimFC(self.fc_size, num_outputs),)
        self.value_branch = nn.Sequential(
            self.fc_1,
            nn.ReLU(),
            SlimFC(self.fc_size, 1),
        )   
        self._features = None

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return th.reshape(self.value_branch(self._features), [-1])

    def forward(self, input_dict, state, seq_lens):
        input = input_dict['obs'].permute(0, 4, 1, 2, 3)
        x = nn.functional.relu(self.conv_1(input.float()))
        x_act = x[:, :, x.shape[2] // 2 - 1: x.shape[2] // 2 + 2, x.shape[3] // 2 - 1: x.shape[3] // 2 + 2, 
                    x.shape[4] // 2 - 1: x.shape[4] // 2 +2].reshape(x.size(0), -1)
        x = x.reshape(x.size(0), -1)
        self._features = x
        action_out = self.action_branch(x_act)

        return action_out, []


class SeqNCA(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                #  n_aux_chan=0,
                 ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        
        custom_model_config = model_config["custom_model_config"]
        # HACK: Because rllib silently squashes our multi-agent observation somewhere along the way??? :D
        obs_space = custom_model_config['dummy_env_obs_space']

        conv_filters = custom_model_config['conv_filters']
        fc_size = custom_model_config['fc_size']

        # self.n_aux_chan = n_aux_chan
        self.conv_filters = conv_filters
        # self.obs_size = get_preprocessor(obs_space)(obs_space).size
        obs_shape = obs_space.shape
        # orig_obs_space = model_config['custom_model_config']['orig_obs_space']
        # obs_shape = orig_obs_space['map'].shape
        # metrics_size = orig_obs_space['ctrl_metrics'].shape \
            # if 'ctrl_metrics' in orig_obs_space.spaces else (0,)
        # assert len(metrics_size) == 1
        # metrics_size = metrics_size[0]
        # self.pre_fc_size = (obs_shape[-2] - 2) * (obs_shape[-3] - 2) * conv_filters + metrics_size
        self.pre_fc_size = (obs_shape[-2] - 2) * (obs_shape[-3] - 2) * conv_filters
        self.fc_size = fc_size

        # TODO: use more convolutions here? Change and check that we can still overfit on binary problem.
        # self.conv_1 = nn.Conv2d(obs_shape[-1] + n_aux_chan, out_channels=conv_filters + n_aux_chan, kernel_size=3, stride=1, padding=0)
        self.conv_1 = nn.Conv2d(obs_shape[-1], out_channels=conv_filters, kernel_size=3, stride=1, padding=0)

        self.fc_1 = SlimFC(self.pre_fc_size, self.fc_size)
        self.action_branch = nn.Sequential(
            # SlimFC(3 * 3 * conv_filters + metrics_size, self.fc_size),
            SlimFC(3 * 3 * conv_filters, self.fc_size),
            nn.ReLU(),
            SlimFC(self.fc_size, num_outputs),)
        self.value_branch = nn.Sequential(
            self.fc_1,
            nn.ReLU(),
            SlimFC(self.fc_size, 1),
        )   
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return th.reshape(self.value_branch(self._features), [-1])

    def forward(self, input_dict, state, seq_lens):
        input = input_dict['obs'].permute(0, 3, 1, 2)
        # input = th.cat([input, self._last_aux_activ], dim=1)
        x = nn.functional.relu(self.conv_1(input.float()))
        #NOTE: assuming that the input is padded, and centered at the agent!
        x_act = x[:, :, x.shape[2] // 2 - 1: x.shape[2] // 2 + 2, x.shape[3] // 2 - 1: x.shape[3] // 2 + 2].reshape(x.size(0), -1)
        x = x.reshape(x.size(0), -1)

        # input = input_dict['obs']
        # map = input['map'].permute(0, 3, 1, 2)  # Because rllib order tensors the tensorflow way (channel last)
        # x = nn.functional.relu(self.conv_1(map.float()))
        # ctrl_metrics = input['ctrl_metrics']
        # #NOTE: assuming that the input is padded, and centered at the agent!
        # x_act = th.cat((x[:, :, x.shape[2] // 2 - 1:x.shape[2] // 2 + 2, x.shape[3] // 2 - 1:x.shape[3] // 2 + 2].reshape(x.size(0), -1), ctrl_metrics), dim=1)
        # x = th.cat((x.reshape(x.size(0), -1), ctrl_metrics), dim=1)
        self._features = x
        action_out = self.action_branch(x_act)

        return action_out, []


class ConvDeconv2d(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 conv_filters=64,
                 fc_size=64,
                #  n_aux_chan=0,
                 ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        # self.n_aux_chan = n_aux_chan
        self.conv_filters = conv_filters
        # self.obs_size = get_preprocessor(obs_space)(obs_space).size
        obs_shape = obs_space.shape
        # orig_obs_space = model_config['custom_model_config']['orig_obs_space']
        # obs_shape = orig_obs_space['map'].shape
        # metrics_size = orig_obs_space['ctrl_metrics'].shape \
            # if 'ctrl_metrics' in orig_obs_space.spaces else (0,)
        # assert len(metrics_size) == 1
        # metrics_size = metrics_size[0]
        # self.pre_fc_size = (obs_shape[-2] - 2) * (obs_shape[-3] - 2) * conv_filters + metrics_size
        # self.pre_fc_size = (obs_shape[-2] - 2) * (obs_shape[-3] - 2) * conv_filters
        self.fc_size = fc_size

        # TODO: use more convolutions here? Change and check that we can still overfit on binary problem.
        # self.conv_1 = nn.Conv2d(obs_shape[-1] + n_aux_chan, out_channels=conv_filters + n_aux_chan, kernel_size=3, stride=1, padding=0)
        self.conv_1 = nn.Conv2d(obs_shape[-1], out_channels=conv_filters, kernel_size=7, stride=2, padding=3)
        self.conv_2 = nn.Conv2d(conv_filters, out_channels=conv_filters, kernel_size=7, stride=2, padding=3)
        self.deconv_1 = nn.ConvTranspose2d(conv_filters, conv_filters, kernel_size=7, stride=1, padding=3)
        n_actions = int(num_outputs / (obs_shape[-2] * obs_shape[-3]))
        self.deconv_2 = nn.ConvTranspose2d(conv_filters, n_actions, kernel_size=7, stride=2, padding=0)
        dummy_pre_fc = self.conv_2(self.conv_1(th.zeros(1, obs_shape[-1], *obs_shape[:-1])))
        pre_fc_shape = dummy_pre_fc.shape
        pre_fc_size = dummy_pre_fc.view(1, -1).shape[1]

        self.fc_1 = SlimFC(pre_fc_size, pre_fc_size)
        # self.action_branch = nn.Sequential(
        #     # SlimFC(3 * 3 * conv_filters + metrics_size, self.fc_size),
        #     SlimFC(3 * 3 * conv_filters, self.fc_size),
        #     nn.ReLU(),
        #     SlimFC(self.fc_size, num_outputs),)
        self.value_branch = nn.Sequential(
            self.fc_1,
            nn.ReLU(),
            SlimFC(pre_fc_size, 1),
        )   
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return th.reshape(self.value_branch(self._features), [-1])

    def forward(self, input_dict, state, seq_lens):
        input = input_dict['obs'].permute(0, 3, 1, 2)
        # input = th.cat([input, self._last_aux_activ], dim=1)
        x1 = nn.functional.relu(self.conv_1(input))
        x2 = nn.functional.relu(self.conv_2(x1))
        pre_fc_shape = x2.shape
        x = x2.reshape(x2.size(0), -1)
        x = self.fc_1(x)
        self._features = x
        x = x.reshape(*pre_fc_shape)
        x = nn.functional.relu(self.deconv_1(x)) 
        x = x + x1
        x = nn.functional.relu(self.deconv_2(x))
        action_out = x.reshape(x.size(0), -1)

        return action_out, []


class CustomFeedForwardModel3D(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 conv_filters=64,
                 fc_size=128,
                 ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        # self.obs_size = get_preprocessor(obs_space)(obs_space).size
        obs_shape = obs_space.shape

        # Determine size of activation after convolutional layers so that we can initialize the fully-connected layer 
        # with the correct number of weights.
        # TODO: figure this out properly, independent of map size. Here we just assume width/height/length of 
        # (padded) observation is 14
        # self.pre_fc_size = (obs_shape[-2] - 2) * (obs_shape[-3] - 2) * 32
        # self.pre_fc_size = 64 * obs_shape[-2] * obs_shape[-3] * obs_shape[-4]

        # Convolutinal layers.
        self.conv_1 = nn.Conv3d(obs_space.shape[-1], out_channels=conv_filters, kernel_size=7, stride=2, padding=1)  # 7 * 7 * 7
        self.conv_2 = nn.Conv3d(64, out_channels=128, kernel_size=3, stride=2, padding=1)  # 4 * 4 * 4
#       self.conv_3 = nn.Conv3d(128, out_channels=128, kernel_size=3, stride=2, padding=1)  # 2 * 2 * 2

        self.pre_fc_size = self.conv_2(self.conv_1(th.zeros(1, obs_shape[-1], *obs_shape[:-1]))).reshape(1, -1).shape[1]
        # Fully connected layer.
        self.fc_1 = SlimFC(self.pre_fc_size, fc_size)

        # Fully connected action and value heads.
        self.action_branch = SlimFC(fc_size, num_outputs)
        self.value_branch = SlimFC(fc_size, 1)

        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return th.reshape(self.value_branch(self._features), [-1])

    def forward(self, input_dict, state, seq_lens):
        input = input_dict["obs"].permute(0, 4, 1, 2, 3)  # Because rllib order tensors the tensorflow way (channel last)
        x = nn.functional.relu(self.conv_1(input.float()))
        x = nn.functional.relu(self.conv_2(x))
#       x = nn.functional.relu(self.conv_2(x.float()))
#       x = nn.functional.relu(self.conv_3(x.float()))
        x = x.reshape(x.size(0), -1)
        x = nn.functional.relu(self.fc_1(x))
        self._features = x
        action_out = self.action_branch(self._features)

        return action_out, []


class WideModel3D(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 n_hid_filters=64,  # number of "hidden" filters in convolutional layers
                # fc_size=128,
                 ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        # How many possible actions can the agent take *at a given coordinate*.
        num_output_actions = num_outputs // np.prod(obs_space.shape[:-1])

        # self.obs_size = get_preprocessor(obs_space)(obs_space).size
        obs_shape = obs_space.shape

        # Determine size of activation after convolutional layers so that we can initialize the fully-connected layer 
        # with the correct number of weights.
        # TODO: figure this out properly, independent of map size. Here we just assume width/height/length of 
        # (padded) observation is 14
        # self.pre_fc_size = (obs_shape[-2] - 2) * (obs_shape[-3] - 2) * 32
        # self.pre_fc_size = 128 * 2 * 2 * 2

        # Size of activation after flattening, after convolutional layers and before the value branch.
        pre_val_size = (obs_shape[-2]) * (obs_shape[-3]) * (obs_shape[-4]) * num_output_actions

        # Convolutinal layers.
        self.conv_1 = nn.Conv3d(obs_space.shape[-1], out_channels=n_hid_filters, kernel_size=5, padding=2)  # 64 * 7 * 7 * 7   
        self.conv_2 = nn.Conv3d(n_hid_filters, out_channels=n_hid_filters, kernel_size=5, padding=2)  # 64 * 7 * 7 * 7
        self.conv_3 = nn.Conv3d(n_hid_filters, out_channels=n_hid_filters, kernel_size=5, padding=2)  # 64 * 7 * 7 * 7
#       self.conv_4 = nn.Conv3d(n_hid_filters, out_channels=n_hid_filters, kernel_size=5, padding=2)  # 64 * 7 * 7 * 7
#       self.conv_5 = nn.Conv3d(n_hid_filters, out_channels=n_hid_filters, kernel_size=3, padding=1)  # 64 * 7 * 7 * 7
#       self.conv_6 = nn.Conv3d(n_hid_filters, out_channels=n_hid_filters, kernel_size=3, padding=1)  # 64 * 7 * 7 * 7
#       self.conv_7 = nn.Conv3d(n_hid_filters, out_channels=n_hid_filters, kernel_size=3, padding=1)  # 64 * 7 * 7 * 7
        self.conv_8 = nn.Conv3d(n_hid_filters, out_channels=num_output_actions, kernel_size=5, padding=2)  # 64 * 7 * 7 * 7 

        # Fully connected layer.
        # self.fc_1 = SlimFC(self.pre_fc_size, fc_size)

        # Fully connected action and value heads.
        # self.action_branch = SlimFC(fc_size, num_outputs)
        self.value_branch = SlimFC(pre_val_size, 1)

        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return th.reshape(self.value_branch(self._features), [-1])

    def forward(self, input_dict, state, seq_lens):
        # Because rllib order tensors the tensorflow way (channel last), we swap the order of the tensor to comply with
        # pytorch.
        input = input_dict["obs"].permute(0, 4, 1, 2, 3)

        x = nn.functional.relu(self.conv_1(input.float()))
        x = nn.functional.relu(self.conv_2(x.float()))
        x = nn.functional.relu(self.conv_3(x.float()))
#       x = nn.functional.relu(self.conv_4(x.float()))
#       x = nn.functional.relu(self.conv_5(x.float()))
#       x = nn.functional.relu(self.conv_6(x.float()))
#       x = nn.functional.relu(self.conv_7(x.float()))
        x = nn.functional.relu(self.conv_8(x.float()))

        # So that we flatten in a way that matches the dimensions of the observation space.
        x = x.permute(0, 2, 3, 4, 1)

        # Flatten the tensor
        x = x.reshape(x.size(0), -1)

        self._features = x
        action_out = x

        return action_out, []


class WideModel3DSkip(WideModel3D, nn.Module):
    def forward(self, input_dict, state, seq_lens):
        input = input_dict["obs"].permute(0, 4, 1, 2, 3)  # Because rllib order tensors the tensorflow way (channel last)
        x1 = nn.functional.relu(self.conv_1(input.float()))
        x2 = nn.functional.relu(self.conv_2(x1.float())) 
        x3 = nn.functional.relu(self.conv_3(x2.float())) + x2
        # x4 = nn.functional.relu(self.conv_4(x3.float()))

        # x5 = nn.functional.relu(self.conv_5(x4.float())) + x4
        # x6 = nn.functional.relu(self.conv_6(x5.float())) + x3
        # x7 = nn.functional.relu(self.conv_7(x6.float())) + x2
        x8 = nn.functional.relu(self.conv_8(x3.float()))

        # So that we flatten in a way that matches the dimensions of the observation space.
        x = x8.permute(0, 2, 3, 4, 1)

        x = x.reshape(x.size(0), -1)
        self._features = x
        action_out = x

        return action_out, []


def init_weights(m):
    if type(m) == th.nn.Linear:
        th.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

    if type(m) == th.nn.Conv2d:
        th.nn.init.orthogonal_(m.weight)



class NCA(TorchModelV2, nn.Module):
    """ A neural cellular automata-type NN to generate levels or wide-representation action distributions."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        conv_filters = model_config.get('custom_model_config').get('conv_filters', 128)
        n_hid_1 = n_hid_2 = conv_filters
        # n_hid_1 = 128
        # n_hid_2 = 128
        n_in_chans = obs_space.shape[-1]
        # TODO: have these supplied to `__init__`
        # n_out_chans = n_in_chans - 1  # assuming we observa path
        n_out_chans = n_in_chans
        w_out = obs_space.shape[0]  # assuming no observable border
        h_out = obs_space.shape[1]

        self.l1 = Conv2d(n_in_chans + 2, n_hid_1, 3, 1, 1, bias=True)  # +2 for x, y coordinates at each tile
        self.l2 = Conv2d(n_hid_1, n_hid_1, 1, 1, 0, bias=True)
        self.l3 = Conv2d(n_hid_1, n_out_chans, 1, 1, 0, bias=True)
        # self.l_vars = nn.Conv2d(n_hid_1, n_out_chans, 1, 1, 0, bias=True)
        # self.l3 = Conv2d(n_hid_1, n_out_chans * 2, 1, 1, 0, bias=True)
        self.value_branch = SlimFC(n_out_chans * w_out * h_out, 1)
        # self.value_branch = SlimFC(n_out_chans * w_out * h_out * 2, 1)
        # self.layers = [self.l1, self.l2, self.l3]
        with th.no_grad():
            self.indices = (th.Tensor(np.indices((w_out, h_out)))[None,...] / max(w_out, h_out)) * 2 - 1
            print('indices max:', self.indices.max(), 'indices min:', self.indices.min())
        self.apply(init_weights)

    def forward(self, input_dict, state, seq_lens):
        x0 = input_dict["obs"].permute(0, 3, 1, 2)  # Because rllib order tensors the tensorflow way (channel last)
        x = th.cat([x0, th.tile(self.indices.to(device=x0.device), (x0.shape[0], 1, 1, 1))], dim=1)
        x = self.l1(x)
        x = th.relu(x)
        x = self.l2(x)
        x = th.relu(x)
        # vars = self.l_vars(x)
        x = self.l3(x)
        x = th.relu(x)
        # x = th.softmax(x, dim=1)
        # mask = th.rand(size=(x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype) < 0.1
        # x = x * mask
        # vars = vars * mask
        # x = 0.5 * x + 0.5 * x0[:,:2] * ~mask  # assume binary, maybe observable path
        # x = x * mask + x0 * ~mask  # assume binary, maybe observable path

        # So that we flatten in a way that matches the dimensions of the observation space.
        x = x.permute(0, 2, 3, 1)

        x = x.reshape(x.size(0), -1)
        # vars = vars.reshape(vars.size(0), -1) - 5

        # x = x0[:, :2].reshape(x.size(0), -1)
        # vars = th.empty_like(x).fill_(0)
        # x = th.cat([x, vars], dim=1)

        self._features = x
        # x = x0[:, :2].reshape(x.size(0), -1)
        # vars = th.empty_like(x).fill_(0)
        # x = th.cat([x, vars], dim=1)


        return x, []

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        vals = th.reshape(self.value_branch(self._features), [-1])
        return vals


class DenseNCA(TorchModelV2, nn.Module):
    """ A neural cellular automata-type NN to generate levels or wide-representation action distributions."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        # conv_filters = model_config.get('custom_model_config').get('conv_filters', 128)
        fc_size = model_config.get('custom_model_config').get('fc_size', 128)
        # n_hid_1 = 128
        # n_hid_2 = 128
        n_in_chans = obs_space.shape[-1]
        # TODO: have these supplied to `__init__`
        # n_out_chans = n_in_chans - 1  # assuming we observa path
        n_out_chans = n_in_chans
        w_out = w_in = obs_space.shape[0]  # assuming no observable border
        h_out = h_in = obs_space.shape[1]

        self.l1 = SlimFC(n_in_chans * w_in * h_in, fc_size)
        self.l2 = SlimFC(fc_size, fc_size)
        self.l3 = SlimFC(fc_size, n_out_chans * w_out * h_out)
        # self.l_vars = nn.Conv2d(n_hid_1, n_out_chans, 1, 1, 0, bias=True)
        # self.l3 = Conv2d(n_hid_1, n_out_chans * 2, 1, 1, 0, bias=True)
        # self.value_branch = SlimFC(n_out_chans * w_out * h_out, 1)
        self.value_branch = SlimFC(n_out_chans * w_out * h_out, 1)
        # self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)

    def forward(self, input_dict, state, seq_lens):
        x0 = input_dict["obs"].permute(0, 3, 1, 2)  # Because rllib order tensors the tensorflow way (channel last)
        x = x0.reshape(x0.size(0), -1)
        with th.no_grad():
            x.fill_(1.0)
        x = self.l1(x)
        x = th.relu(x)
        x = self.l2(x)
        x = th.relu(x)
        # vars = self.l_vars(x)
        x = self.l3(x)
        x = th.relu(x)
        # x = th.softmax(x, dim=1)
        # mask = th.rand(size=(x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype) < 0.1
        # x = x * mask
        # vars = vars * mask
        # x = 0.5 * x + 0.5 * x0[:,:2] * ~mask  # assume binary, maybe observable path
        # x = x * mask + x0 * ~mask  # assume binary, maybe observable path

        # So that we flatten in a way that matches the dimensions of the observation space.
        # x = x.permute(0, 2, 3, 1)

        # x = x.reshape(x.size(0), -1)
        # vars = vars.reshape(vars.size(0), -1) - 5

        self._features = x
        # x = x0[:, :2].reshape(x.size(0), -1)
        vars = th.empty_like(x).fill_(0)
        x = th.cat([x, vars], dim=1)


        # axis 0 is batch
        # axis 1 is the tile-type (one-hot)
        # axis 0,1 is the x value
        # axis 0,2 is the y value

        return x, []

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        vals = th.reshape(self.value_branch(self._features), [-1])
        return vals
    

class Decoder(TorchModelV2, nn.Module):
    """ A neural cellular automata-type NN to generate levels or wide-representation action distributions."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        # conv_filters = model_config.get('custom_model_config').get('conv_filters', 128)
        fc_size = model_config.get('custom_model_config').get('fc_size', 128)
        # n_hid_1 = 128
        # n_hid_2 = 128
        n_in_chans = obs_space.shape[-1]
        # TODO: have these supplied to `__init__`
        # n_out_chans = n_in_chans - 1  # assuming we observa path
        n_out_chans = n_in_chans
        self.w_out = w_out = w_in = obs_space.shape[0]  # assuming no observable border
        self.h_out = h_out = h_in = obs_space.shape[1]

        # self.l1 = SlimFC(n_in_chans * w_in * h_in, fc_size)
        # self.l2 = SlimFC(fc_size, fc_size)
        # self.l3 = SlimFC(fc_size, n_out_chans * w_out * h_out)
        self.l1 = SlimFC(1, 2 * n_out_chans * w_out * h_out)
        # self.l1 = SlimFC(1, 64 * w_out * h_out)
        # self.l_vars = nn.Conv2d(n_hid_1, n_out_chans, 1, 1, 0, bias=True)
        # self.l2 = Conv2d(64, 64, 3, 1, 1, bias=True)
        # self.l3 = Conv2d(64, 2 * n_out_chans, 1, 1, 0, bias=True)
        # self.value_branch = SlimFC(n_out_chans * w_out * h_out, 1)
        self.value_branch = SlimFC(n_out_chans * w_out * h_out, 1)
        # self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)

    def forward(self, input_dict, state, seq_lens):
        x0 = input_dict["obs"].permute(0, 3, 1, 2)  # Because rllib order tensors the tensorflow way (channel last)
        x = x0.reshape(x0.size(0), -1)
        x = th.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        x = self.l1(x)
        # x = x.reshape(x.size(0), 64, self.w_out, self.h_out)
        # for _ in range(10):
            # x = self.l2(x)
            # x = th.relu(x)
        # x = self.l3(x)
        x = x.reshape(x.size(0), 2, -1)
        vars = x[:, 0]
        x = th.relu(x[:, 1])
        # x = th.relu(x)
        # x = self.l2(x)
        # x = th.relu(x)
        # vars = self.l_vars(x)
        # x = self.l3(x)
        # x = th.relu(x)
        # x = th.softmax(x, dim=1)
        # mask = th.rand(size=(x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype) < 0.1
        # x = x * mask
        # vars = vars * mask
        # x = 0.5 * x + 0.5 * x0[:,:2] * ~mask  # assume binary, maybe observable path
        # x = x * mask + x0 * ~mask  # assume binary, maybe observable path

        # So that we flatten in a way that matches the dimensions of the observation space.
        # x = x.permute(0, 2, 3, 1)

        # x = x.reshape(x.size(0), -1)
        # vars = vars.reshape(vars.size(0), -1) - 5

        self._features = x
        # x = x0[:, :2].reshape(x.size(0), -1)
        # vars = th.empty_like(x).fill_(0)
        x = th.cat([x, vars], dim=1)


        # axis 0 is batch
        # axis 1 is the tile-type (one-hot)
        # axis 0,1 is the x value
        # axis 0,2 is the y value

        return x, []

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        vals = th.reshape(self.value_branch(self._features), [-1])
        return vals

