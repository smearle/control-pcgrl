#!/usr/bin/env python3
#        This file is part of qdpy.
#
#        qdpy is free software: you can redistribute it and/or modify
#        it under the terms of the GNU Lesser General Public License as
#        published by the Free Software Foundation, either version 3 of
#        the License, or (at your option) any later version.
#
#        qdpy is distributed in the hope that it will be useful,
#        but WITHOUT ANY WARRANTY; without even the implied warranty of
#        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#        GNU Lesser General Public License for more details.
#
#        You should have received a copy of the GNU Lesser General Public
#        License along with qdpy. If not, see <http://www.gnu.org/licenses/>.


########## IMPORTS ########### {{{1

# qd
import qdpy
from qdpy.base import *
from qdpy.experiment import QDExperiment

# bipedal
from sim import Model, simulate, make_env
from bipedal_walker import BipedalWalkerExperiment

# Pytorch
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

#from timeit import default_timer as timer


########## PyTorch ########### {{{1

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.tanh = nn.Tanh()
        self.hidden = []
        for i in range(1, len(hidden_sizes)):
            self.hidden.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.fc2 = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        for hidden in self.hidden:
            out = hidden(out)
            out = self.tanh(out)
        out = self.fc2(out)
        return out


class PyTorchModel(Model):
    ''' Simple MLP implemented by PyTorch '''
    def __init__(self, config):
        super().__init__(config)

    def init_nn(self):
        self.nn = MLP(self.input_size, self.layers, self.output_size)
        params = parameters_to_vector(self.nn.parameters())
        self.param_count = len(params)

    def get_action(self, x):
        inputs = torch.Tensor(x)
        return self.nn.forward(inputs).detach().numpy()

    def set_model_params(self, model_params):
        #start_time = timer()
        params = parameters_to_vector(self.nn.parameters())
        params[:] = torch.from_numpy(np.array(model_params))
        vector_to_parameters(params, self.nn.parameters())
        #print(f"DEBUG set_model_params duration: {timer() - start_time}")



########## EXPERIMENT CLASS ########### {{{1

class BipedalWalkerPyTorchExperiment(BipedalWalkerExperiment):
    def init_model(self):
        self.model = PyTorchModel(self.config['game'])


########## BASE FUNCTIONS ########### {{{1

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configFilename', type=str, default='conf/test.yaml', help = "Path of configuration file")
    parser.add_argument('-o', '--resultsBaseDir', type=str, default='results/', help = "Path of results files")
    parser.add_argument('-p', '--parallelismType', type=str, default='concurrent', help = "Type of parallelism to use")
    parser.add_argument('--replayBestFrom', type=str, default='', help = "Path of results data file -- used to replay the best individual")
    parser.add_argument('--seed', type=int, default=None, help="Numpy random seed")
    return parser.parse_args()

def create_base_config(args):
    base_config = {}
    if len(args.resultsBaseDir) > 0:
        base_config['resultsBaseDir'] = args.resultsBaseDir
    return base_config

def create_experiment(args, base_config):
    exp = BipedalWalkerPyTorchExperiment(args.configFilename, args.parallelismType, seed=args.seed, base_config=base_config)
    print("Using configuration file '%s'. Instance name: '%s'" % (args.configFilename, exp.instance_name))
    return exp

def launch_experiment(exp):
    exp.run()

def replay_best(args, exp):
    import pickle
    path = args.replayBestFrom
    with open(path, "rb") as f:
        data = pickle.load(f)
    best = data['container'].best
    exp.eval_fn(best, render_mode = True)



########## MAIN ########### {{{1
if __name__ == "__main__":
    import traceback
    args = parse_args()
    base_config = create_base_config(args)
    try:
        exp = create_experiment(args, base_config)
        if len(args.replayBestFrom) > 0:
            replay_best(args, exp)
        else:
            launch_experiment(exp)
    except Exception as e:
        warnings.warn(f"Run failed: {str(e)}")
        traceback.print_exc()


# MODELINE  "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
