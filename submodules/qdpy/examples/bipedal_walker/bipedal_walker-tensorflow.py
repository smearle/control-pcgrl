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
import evo.models
from sim import Model, simulate, make_env
from bipedal_walker import BipedalWalkerExperiment

# Tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
tf.compat.v1.disable_eager_execution()
#tf.compat.v1.disable_v2_behavior()
tf.config.threading.set_inter_op_parallelism_threads(1)

#from timeit import default_timer as timer


########## KERAS ########### {{{1


class EvolvableKerasModel(object):
    """ Interface to use keras/tf.keras models with evolutionary algorithms """

    def __init__(self, model, nbInputs, nbOutputs):
        self.nbInputs = nbInputs
        self.nbOutputs = nbOutputs
        self.nbWeights = model.count_params()

        #self.architecture = model.to_json()
        #self.model = tf.keras.models.model_from_json(self.architecture)
        self.model = model

    def clone(self):
        return EvolvableKerasModel(self.model, self.nbInputs, self.nbOutputs)

    def reinit(self):
        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        #self.model._make_predict_function()
        modelW = self.model.get_weights()
        self.modelShape = []
        self.modelSize = []
        for i in range(len(modelW)):
            tmpShape = modelW[i].shape
            self.modelShape.append(tmpShape)
            self.modelSize.append(np.prod(tmpShape))

    def clear(self):
        self._inputs = np.zeros(self.nbInputs)
        self._outputs = np.zeros(self.nbOutputs)

    def step(self, nbSteps = 1):
        modelInputs = self._inputs.reshape((1, self.nbInputs))
        for i in range(nbSteps):
            self._outputs = self.model.predict(modelInputs)[0]

    def getNbWeights(self):
        return self.nbWeights

    def setInputs(self, inputs):
        self._inputs = np.array(inputs)

    def outputs(self):
        return self._outputs

    def setWeights(self, weights):
        weightsIndice = 0
        w = np.array(weights)
        modelW = self.model.get_weights()
        for i in range(len(self.modelSize)):
            modelW[i][:] = np.resize(w[weightsIndice:weightsIndice+self.modelSize[i]], (self.modelShape[i]))
            weightsIndice += self.modelSize[i]
        evo.models.set_weights(modelW)


class TFModel(Model):
    ''' Simple MLP implemented by Tensorflow.Keras '''
    def __init__(self, config):
        super().__init__(config)

    def _create_nn(self):
        activation = 'tanh'
        model = tf.keras.Sequential()
        for i, l in enumerate(self.layers):
            input_dim = self.input_size if i == 0 else None
            model.add(tf.keras.layers.Dense(units=l, activation=activation, input_dim=input_dim))
        model.add(tf.keras.layers.Dense(units=self.output_size, activation=activation))
        nn = EvolvableKerasModel(model, self.input_size, self.output_size)
        return nn

    def init_nn(self):
        nn = self._create_nn()
        self.param_count = nn.getNbWeights()

    def get_action(self, x):
        self.nn.setInputs(x)
        self.nn.step()
        return np.array(self.nn.outputs())

    def set_model_params(self, model_params):
        #start_time = timer()
        self.nn = self._create_nn()
        self.nn.reinit()
        self.nn.setWeights(model_params)
        #print(f"DEBUG set_model_params duration: {timer() - start_time}")



########## EXPERIMENT CLASS ########### {{{1

class BipedalWalkerTFExperiment(BipedalWalkerExperiment):
    def init_model(self):
        self.model = TFModel(self.config['game'])


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
    exp = BipedalWalkerTFExperiment(args.configFilename, args.parallelismType, seed=args.seed, base_config=base_config)
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
