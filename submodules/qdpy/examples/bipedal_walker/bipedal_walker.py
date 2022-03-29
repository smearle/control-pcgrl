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


########## EXPERIMENT CLASS ########### {{{1

class BipedalWalkerExperiment(QDExperiment):
    def reinit(self):
        super().reinit()
        self.env_name = self.config['game']['env_name']
        self.init_model()
        self.update_dimension()

    def init_model(self):
        self.model = Model(self.config['game'])

    def update_dimension(self):
        self.algo.dimension = self.model.param_count

    def eval_fn(self, ind, render_mode = False):
        env = make_env(self.env_name)
        self.model.set_model_params(ind)
        scores = simulate(self.model,
                env,
                render_mode=render_mode,
                num_episode=self.config['indv_eps'])
        ind.fitness.values = scores[self.fitness_type],
        ind.features.values = [scores[x] for x in self.features_list]
        return ind


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
    exp = BipedalWalkerExperiment(args.configFilename, args.parallelismType, seed=args.seed, base_config=base_config)
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
