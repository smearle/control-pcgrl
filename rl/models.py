from typing import Dict, List

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
                 ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        # self.obs_size = get_preprocessor(obs_space)(obs_space).size
        obs_shape = obs_space.shape
        self.pre_fc_size = (obs_shape[-2] - 2) * (obs_shape[-3] - 2) * conv_filters
        self.fc_size = fc_size

        # TODO: use more convolutions here? Change and check that we can still overfit on binary problem.
        self.conv_1 = nn.Conv2d(obs_space.shape[-1], out_channels=conv_filters, kernel_size=3, stride=1, padding=0)

        self.fc_1 = SlimFC(self.pre_fc_size, self.fc_size)
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
        x = x.reshape(x.size(0), -1)
        x = nn.functional.relu(self.fc_1(x))
        self._features = x
        action_out = self.action_branch(self._features)

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
        self.pre_fc_size = 64 * obs_shape[-2] * obs_shape[-3] * obs_shape[-4]

        # Convolutinal layers.
        self.conv_1 = nn.Conv3d(obs_space.shape[-1], out_channels=conv_filters, kernel_size=3, stride=1, padding=1)  # 7 * 7 * 7
#       self.conv_2 = nn.Conv3d(64, out_channels=128, kernel_size=3, stride=2, padding=1)  # 4 * 4 * 4
#       self.conv_3 = nn.Conv3d(128, out_channels=128, kernel_size=3, stride=2, padding=1)  # 2 * 2 * 2
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

    def __init__(self, obs_space, action_space, num_outputs, model_config,name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        n_hid_1 = 256
        n_hid_2 = 256
        n_in_chans = obs_space.shape[-1]
        self.l1 = Conv2d(n_in_chans, n_hid_1, 3, 1, 1, bias=True)
        self.l2 = Conv2d(n_hid_1, n_hid_2, 1, 1, 0, bias=True)
        self.l3 = Conv2d(n_hid_2, n_in_chans, 1, 1, 0, bias=True)
        self.value_branch = SlimFC(np.prod(obs_space.shape), 1)
        # self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].permute(0, 3, 1, 2)  # Because rllib order tensors the tensorflow way (channel last)
        x = self.l1(x)
        x = th.relu(x)
        x = self.l2(x)
        x = th.relu(x)
        x = self.l3(x)
        # TODO: try softmax
        x = th.relu(x)

        # So that we flatten in a way that matches the dimensions of the observation space.
        x = x.permute(0, 2, 3, 1)

        x = x.reshape(x.size(0), -1)
        self._features = x

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
    

# NEXT: change action space to allow multiple change at each step 
# NEXT: add path length, connectivity, region num to tensorboard logs
# NEXT: make all 3D envs holey (if we want 3D probs holey, we need to make pcgrl_env_3D a holey env for env register)