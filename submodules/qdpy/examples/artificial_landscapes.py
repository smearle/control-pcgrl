#!/usr/bin/env python3
#    This file is part of qdpy.
#
#    qdpy is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    qdpy is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with qdpy. If not, see <http://www.gnu.org/licenses/>.


"""An presentation of a simple benchmark for grid-based QD algorithms by using artificial landscapes (i.e. test functions for optimisation tasks). In this benchmark, a test function (e.g. the Rastrigin function) is illuminated with features corresponding to the first two parameters of each genome."""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

from qdpy import algorithms, containers, benchmarks, plots

import numpy as np
import warnings
import os
import random
from scipy.constants import golden_ratio


def iteration_callback(algo, batch_elapsed, grid_ref):
    global_reliability = compute_global_reliability(grid_ref, algo.container)
    algo.container.global_reliability.append(global_reliability)
    #print(f"global_reliability = {global_reliability}")

def tell_callback(algo, ind, grid_ref):
    global_reliability = compute_global_reliability(grid_ref, algo.container)
    algo.container.global_reliability.append(global_reliability)
    #print(f"global_reliability = {global_reliability}")

def cleanup_data(a): # Assume minimisation
    a2 = a.copy()
    a2[np.isinf(a2)] = np.nan
    return a2

def normalise(a, min_val, max_val): # Assume minimisation
    a2 = a.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a2[a2>max_val] = np.nan
        a2[a2<min_val] = min_val
    return (a2-min_val)/(max_val-min_val)

def compute_global_reliability(grid_ref, grid_test): # Assume minimisation
    if isinstance(grid_ref, containers.Grid):
        base_ref = cleanup_data(grid_ref.quality_array[...,0])
    else:
        base_ref = cleanup_data(grid_ref.to_grid(shape=(64,64)).quality_array[...,0]) # XXX
        #base_ref = cleanup_data(np.array([ind.fitness.values[0] for ind in grid_ref])) # XXX

    if isinstance(grid_test, containers.Grid):
        base_test = cleanup_data(grid_test.quality_array[...,0])
    else:
        base_test = cleanup_data(grid_test.to_grid(shape=(64,64)).quality_array[...,0]) # XXX
        #base_test= cleanup_data(np.array([ind.fitness.values[0] for ind in grid_test])) # XXX

    min_ref = np.nanmin(base_ref)
    max_ref = np.nanmax(base_ref)
    #min_ref = min( np.nanmin(base_ref), np.nanmin(base_test) )
    #max_ref = max( np.nanmax(base_ref), np.nanmax(base_test) )

    normalised_ref = normalise(base_ref, min_ref, max_ref)
    normalised_test = normalise(base_test, min_ref, max_ref)
    mask = ~np.isnan(normalised_ref)
    data_ref = normalised_ref[mask]
    data_test = normalised_test[mask]
    #print(data_ref)
    #print(data_test)
    #sqerrors = np.nan_to_num(1. - np.square(data_ref - data_test, dtype=float))
    #sqerrors[sqerrors < 0.0] = 0.
    ##print(sqerrors)
    #global_reliability = np.sum(sqerrors) / len(data_ref)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        local_reliability = np.nan_to_num((1.-data_test) / (1.-data_ref))
    local_reliability[local_reliability<0.0] = 0.
    local_reliability[local_reliability>1.0] = 1.
    global_reliability = np.sum(local_reliability) / len(data_ref)

    return global_reliability


def compute_ref(bench, budget=60000, dimension=2, nb_bins_per_feature=64, output_path="ref", algo_name=None, fitness_domain=((0., 120.),)):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create grid and algorithm
    grid_ref = containers.Grid(shape=(nb_bins_per_feature,) * bench.nb_features, max_items_per_bin=1,
            fitness_domain=bench.fitness_domain, features_domain=bench.features_domain)
    algo_ref = algorithms.RandomSearchMutPolyBounded(grid_ref, budget=budget, batch_size=500,
            dimension=dimension, optimisation_task=bench.default_task, ind_domain=bench.ind_domain,
            name=algo_name)

    # Create a logger to pretty-print everything and generate output data files
    logger_ref = algorithms.TQDMAlgorithmLogger(algo_ref, log_base_path=output_path)
    # Define evaluation function
    eval_fn = bench.fn
    # Run illumination process !
    best = algo_ref.optimise(eval_fn)
    # Print results info
    #print(algo_ref.summary())
    # Plot the results
    plots.default_plots_grid(logger_ref, to_grid_parameters={'shape': (nb_bins_per_feature,) * bench.nb_features}, fitness_domain=fitness_domain)
    return algo_ref, logger_ref


def compute_test(bench, algo_ref, dimension=3, output_path="test", algo_name=None, mut_pb=0.5, eta=20., fitness_domain=((0., 120.),)):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create container and algorithm
    grid_test = containers.Grid(shape=(algo_ref.container.shape[0],) * bench.nb_features, max_items_per_bin=1,
            fitness_domain=bench.fitness_domain, features_domain=bench.features_domain)
    #grid_test = containers.NoveltyArchive(k=1, threshold_novelty=0.016, fitness_domain=bench.fitness_domain, features_domain=bench.features_domain, storage_type=list, depot_type=list)
    grid_test.global_reliability = []
    #algo_test = algorithms.RandomSearchMutPolyBounded(grid_test, budget=algo_ref.budget, batch_size=500,
    #        dimension=dimension, optimisation_task=bench.default_task, ind_domain=bench.ind_domain,
    #        name=algo_name)
    algo_test = algorithms.MutPolyBounded(grid_test, budget=algo_ref.budget, batch_size=500,
            dimension=dimension, optimisation_task=bench.default_task, ind_domain=bench.ind_domain,
            #sel_pb = 1.0, init_pb = 0.0, mut_pb = 0.8, eta = 20., name=algo_name)
            mut_pb = mut_pb, eta = eta, name=algo_name)

    #grid_surrogate = containers.Grid(shape=(algo_ref.container.shape[0],) * bench.nb_features, max_items_per_bin=1,
    #        fitness_domain=bench.fitness_domain, features_domain=bench.features_domain)
    #grid_test = containers.Grid(shape=(algo_ref.container.shape[0],) * bench.nb_features, max_items_per_bin=1,
    #        fitness_domain=bench.fitness_domain, features_domain=bench.features_domain)
    #grid_test.global_reliability = []
    #algo_surrogate = algorithms.RandomSearchMutPolyBounded(grid_surrogate, budget=5000, batch_size=500,
    #        dimension=dimension, optimisation_task=bench.default_task, ind_domain=bench.ind_domain)
    #algo_illumination = algorithms.RandomSearchMutPolyBounded(grid_test, budget=algo_ref.budget, batch_size=500,
    #        dimension=dimension, optimisation_task=bench.default_task, ind_domain=bench.ind_domain)
    #algo_test = algorithms.SAIL(illumination_algo=algo_illumination, acquisition_algo=algo_surrogate, name=algo_name)



    algo_test.add_callback("tell", algorithms.partial(tell_callback, grid_ref=algo_ref.container))
    #algo_test.add_callback("iteration", algorithms.partial(iteration_callback, grid_ref=algo_ref.container))

    # Create a logger to pretty-print everything and generate output data files
    logger_test = algorithms.TQDMAlgorithmLogger(algo_test, log_base_path=output_path)
    # Define evaluation function
    eval_fn = bench.fn
    # Run illumination process !
    best = algo_test.optimise(eval_fn)
    # Print results info
    #print(algo_test.summary())
    # Plot the results
    plots.default_plots_grid(logger_test, to_grid_parameters={'shape': algo_ref.container.shape}, fitness_domain=fitness_domain)
    # Plot global_reliability per eval
    #global_reliability = compute_global_reliability(grid_ref, grid_test)
    #print(f"global_reliability = {global_reliability}")
    plots.plot_evals(grid_test.global_reliability, os.path.join(logger_test.log_base_path, "global_reliability.pdf"), "global_reliability", ylim=(0., 1.))
    print(f"dimension={dimension}  global_reliability[-1]:", grid_test.global_reliability[-1])
    return algo_test, logger_test



def compute_test2(bench, algo_ref, dimension=3, output_path="test", algo_name=None, mut_pb=0.5, mu=0., sigma=1.0, fitness_domain=((0., 120.),)):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create container and algorithm
    grid_test = containers.Grid(shape=(algo_ref.container.shape[0],) * bench.nb_features, max_items_per_bin=1,
            fitness_domain=bench.fitness_domain, features_domain=bench.features_domain)
    grid_test.global_reliability = []
    algo_test = algorithms.MutGaussian(grid_test, budget=algo_ref.budget, batch_size=500,
            dimension=dimension, optimisation_task=bench.default_task, ind_domain=bench.ind_domain,
            mut_pb = mut_pb, mu = mu, sigma=sigma, name=algo_name)

    algo_test.add_callback("tell", algorithms.partial(tell_callback, grid_ref=algo_ref.container))
    #algo_test.add_callback("iteration", algorithms.partial(iteration_callback, grid_ref=algo_ref.container))

    # Create a logger to pretty-print everything and generate output data files
    logger_test = algorithms.TQDMAlgorithmLogger(algo_test, log_base_path=output_path)
    # Define evaluation function
    eval_fn = bench.fn
    # Run illumination process !
    best = algo_test.optimise(eval_fn)
    # Print results info
    #print(algo_test.summary())
    # Plot the results
    plots.default_plots_grid(logger_test, to_grid_parameters={'shape': algo_ref.container.shape}, fitness_domain=fitness_domain)
    # Plot global_reliability per eval
    plots.plot_evals(grid_test.global_reliability, os.path.join(logger_test.log_base_path, "global_reliability.pdf"), "global_reliability", ylim=(0., 1.))
    print(f"dimension={dimension}  global_reliability[-1]:", grid_test.global_reliability[-1])
    return algo_test, logger_test


from collections import OrderedDict
_linestyles = OrderedDict(
    [('solid',               (0, ())),
     #('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     #('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     #('loosely dashdotted',  (0, (3, 10, 1, 10))),
     #('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     #('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])


def plot_combined_global_reliability(algo_ref, algos, output_filename="global_reliability.pdf", figsize=(4.*golden_ratio,4.)):
    assert(len(algos))
    data_tests = [a.container.global_reliability for a in algos]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(data_tests[0]))

    ##linestyle_cycler = cycler('linestyle',['-','--',':','-.','-','--',':']) + cycler(color=plt.get_cmap("Set2",8).colors)
    ##linestyle_cycler = cycler('linestyle', list(_linestyles.values())[:8]) + cycler(color=plt.get_cmap("Dark2",8).colors)
    #linestyle_cycler = cycler('linestyle', list(_linestyles.values())[:8]) + cycler(color=['r', 'r', 'r', 'r', 'g', 'g', 'g', 'g'])
    #linestyle_cycler = cycler('linestyle', list(_linestyles.values())[:4] * 2) + cycler(color=['r', 'g', 'r', 'g', 'r', 'g', 'r', 'g'])
    #linestyle_cycler = cycler('linestyle',['-','-','-','-',':',':',':',':']) + cycler(color=["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3"])
    linestyle_cycler = cycler('linestyle',['-','-','-','-',':',':',':',':']) + cycler(color=["#e66101", "#fdb863", "#b2abd2", "#5e3c99", "#e66101", "#fdb863", "#b2abd2", "#5e3c99"])
    ax.set_prop_cycle(linestyle_cycler)
    plt.xticks(rotation=20)
    for d, a in zip(data_tests, algos):
        ax.plot(x, d, label=a.name, linewidth=3)

    ax.set_ylim((0., 1.))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0E'))
    plt.xlabel("Evaluations", fontsize=20)
    plt.ylabel("Global reliability", fontsize=20)
    for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(19)
    for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(19)

    #plt.tight_layout()
    #plt.legend(title="Dimension", loc="lower right", fontsize=12, title_fontsize=14)
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc="center left", fontsize=16, title_fontsize=16, bbox_to_anchor=(1.04, 0.5), borderaxespad=0)

    fig.savefig(output_filename, bbox_inches="tight")
    plt.close(fig)


def plot3D(bench, output_filename="plot3D.pdf", step=0.1):
    def fn_arg0(ind):
        return bench.fn(ind)[0][0]

    fig = plt.figure(figsize=(4.*golden_ratio,4.))
    ax = fig.add_subplot(111, projection='3d', azim=-19, elev=30, position=[0.25, 0.15, 0.7, 0.7]) 
    X = np.arange(bench.ind_domain[0], bench.ind_domain[1], step)
    Y = np.arange(bench.ind_domain[0], bench.ind_domain[1], step)
    X, Y = np.meshgrid(X, Y)
    Z = np.fromiter(map(fn_arg0, zip(X.flat,Y.flat)), dtype=np.float, count=X.shape[0]*X.shape[1]).reshape(X.shape)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap("inferno_r"), linewidth=0.2)

    ax.set_xlabel("x0", fontsize=14)
    ax.set_ylabel("x1", fontsize=14)
    #ax.set_xlabel("Feature 1", fontsize=14)
    #ax.set_ylabel("Feature 2", fontsize=14)
    ax.set_zlabel("Fitness", fontsize=14)
    # change fontsize
    for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(14)
    for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(14)
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(14)
    plt.tight_layout()
    #fig.subplots_adjust(right=0.85, bottom=0.10, wspace=0.10)
    fig.savefig(output_filename)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help="Numpy random seed")
    parser.add_argument('-p', '--parallelismType', type=str, default='none', help = "Type of parallelism to use (none, concurrent, scoop)")
    parser.add_argument('-o', '--outputDir', type=str, default="results", help = "Path of the output log files")
    parser.add_argument('--bench', type=str, default="rastrigin", help = "Benchmark function to use")
    args = parser.parse_args()

    # Find random seed
    if args.seed is not None:
        seed = args.seed
    else:
        seed = np.random.randint(1000000)

    # Update and print seed
    np.random.seed(seed)
    random.seed(seed)
    print("Seed: %i" % seed)

    # Find where to put logs
    log_base_path = args.outputDir

    # Create container and algorithm. Here we use MAP-Elites, by illuminating a Grid container by evo.
    sigma=1.0
    step=0.1
    if args.bench == "rastrigin": #
        bench = benchmarks.RastriginBenchmark(nb_features=2)
        fitness_domain = ((0., 120.),)
    elif args.bench == "normalisedRastrigin":
        bench = benchmarks.NormalisedRastriginBenchmark(nb_features=2)
        fitness_domain = ((-np.inf, np.inf),)
    elif args.bench == "sphere":
        bench = benchmarks.SphereBenchmark(nb_features=2)
        fitness_domain = ((-np.inf, np.inf),)
    elif args.bench == "eightedSphere":
        bench = benchmarks.WeightedSphereBenchmark(nb_features=2)
        fitness_domain = ((-np.inf, np.inf),)
    elif args.bench == "rotatedHyperEllipsoid":
        bench = benchmarks.RotatedHyperEllipsoidBenchmark(nb_features=2)
        fitness_domain = ((0., 4000.),)
    elif args.bench == "rosenbrock":
        bench = benchmarks.RosenbrockBenchmark(nb_features=2)
        fitness_domain = ((-np.inf, np.inf),)
    elif args.bench == "schwefel":
        bench = benchmarks.SchwefelBenchmark(nb_features=2)
        fitness_domain = ((0., np.inf),)
        step=8.0
    elif args.bench == "small_schwefel":
        bench = benchmarks.SmallSchwefelBenchmark(nb_features=2)
        fitness_domain = ((0., np.inf),)
        step=4.0
    elif args.bench == "griewangk":
        bench = benchmarks.GriewangkBenchmark(nb_features=2)
        fitness_domain = ((-np.inf, np.inf),)
    elif args.bench == "sumOfPowers": #
        bench = benchmarks.SumOfPowersBenchmark(nb_features=2)
        fitness_domain = ((0., 2.),)
    elif args.bench == "ackley":
        bench = benchmarks.AckleyBenchmark(nb_features=2)
        fitness_domain = ((0., 20.),)
    elif args.bench == "styblinskiTang":
        bench = benchmarks.StyblinskiTangBenchmark(nb_features=2)
        fitness_domain = ((-120., 250.),)
    elif args.bench == "levy":
        bench = benchmarks.LevyBenchmark(nb_features=2)
        fitness_domain = ((-np.inf, np.inf),)
    elif args.bench == "perm0db":
        bench = benchmarks.Perm0dbBenchmark(nb_features=2)
        fitness_domain = ((-np.inf, np.inf),)
    elif args.bench == "permdb":
        bench = benchmarks.PermdbBenchmark(nb_features=2)
        fitness_domain = ((-np.inf, np.inf),)
    elif args.bench == "trid":
        bench = benchmarks.TridBenchmark(nb_features=2)
        fitness_domain = ((-np.inf, np.inf),)
    elif args.bench == "zakharov": #
        bench = benchmarks.ZakharovBenchmark(nb_features=2)
        fitness_domain = ((0., 1000.),)
    elif args.bench == "dixonPrice":
        bench = benchmarks.DixonPriceBenchmark(nb_features=2)
        fitness_domain = ((-np.inf, np.inf),)
    elif args.bench == "powell":
        bench = benchmarks.PowellBenchmark(nb_features=2)
        fitness_domain = ((-np.inf, np.inf),)
    elif args.bench == "michalewicz":
        bench = benchmarks.MichalewiczBenchmark(nb_features=2)
        fitness_domain = ((-np.inf, 0.),)
    elif args.bench == "wavy": #
        bench = benchmarks.WavyBenchmark(nb_features=2)
        fitness_domain = ((0., 2.0),)
    elif args.bench == "trigonometric02":
        bench = benchmarks.Trigonometric02Benchmark(nb_features=2)
        fitness_domain = ((1., np.inf),)
        sigma=200.0
    elif args.bench == "qing":
        bench = benchmarks.QingBenchmark(nb_features=2)
        fitness_domain = ((0., np.inf),)
        sigma=200.0
    elif args.bench == "small_qing":
        bench = benchmarks.SmallQingBenchmark(nb_features=2)
        fitness_domain = ((0., 500.),)
        sigma=0.5
    elif args.bench == "deb01":
        bench = benchmarks.Deb01Benchmark(nb_features=2)
        fitness_domain = ((-1., 1.),)
        sigma=1.0
    elif args.bench == "shubert04":
        bench = benchmarks.Shubert04Benchmark(nb_features=2)
        fitness_domain = ((-30., 30.),)
        sigma=2.0
    else:
        raise f"Unknown benchmark '{args.bench}' !"

    #fitness_domain = ((0., np.inf),)

    # Plot 3D
    plot3D(bench, output_filename=os.path.join(log_base_path, "plot3D.pdf"), step=step)

    # Compute reference
    algo_name_ref = args.bench + "-ref"
    algo_ref, logger_ref = compute_ref(bench, budget=1000000, dimension=2, nb_bins_per_feature=64,
    #algo_ref, logger_ref = compute_ref(bench, budget=100000, dimension=2, nb_bins_per_feature=64,
    #algo_ref, logger_ref = compute_ref(bench, budget=1000, dimension=2, nb_bins_per_feature=64,
            output_path=os.path.join(log_base_path, algo_name_ref), algo_name=algo_name_ref, fitness_domain=fitness_domain)

    # Compute benchmark for several dimensions
    #tested_dim = [3, 4, 6, 8, 10, 14]
    tested_dim = [3, 6, 10, 14]
    #tested_dim = [3]
    algos = []
    loggers = []
    for dim in tested_dim:
        #algo_name = f"Dimension {dim}"
        algo_name = f"ME1 {dim} dim"
        a,l = compute_test(bench, algo_ref, dimension=dim,
                output_path=os.path.join(log_base_path, algo_name), algo_name=algo_name, fitness_domain=fitness_domain)
        algos.append(a)
        loggers.append(l)
        #print(a.summary())

    for dim in tested_dim:
        algo_name = f"ME2 {dim} dim"
        a,l = compute_test2(bench, algo_ref, dimension=dim, sigma=sigma,
                output_path=os.path.join(log_base_path, algo_name), algo_name=algo_name, fitness_domain=fitness_domain)
        algos.append(a)
        loggers.append(l)
        #print(a.summary())

    # Make combined plots
    plot_combined_global_reliability(algo_ref, algos, output_filename=os.path.join(log_base_path, "global_reliability.pdf"))


# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
