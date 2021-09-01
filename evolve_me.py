from pdb import set_trace as TT
from qdpy import algorithms, containers, benchmarks, plots
from evolve import get_init_weights

# Create container and algorithm. Here we use MAP-Elites, by illuminating a Grid container by evolution.
grid = containers.Grid(shape=(100,100), max_items_per_bin=1, fitness_domain=((0., 1.),), features_domain=((0., 1.), (0., 1.)))
algo = algorithms.RandomSearchMutPolyBounded(grid, budget=60000, batch_size=500,
        dimension=3, optimisation_task="maximisation")

# Create a logger to pretty-print everything and generate output data files
logger = algorithms.AlgorithmLogger(algo)

# Define evaluation function

#eval_fn = algorithms.partial(benchmarks.illumination_rastrigin_normalised,
#        nb_features = len(grid.shape))
def eval_fn(*args):

        TT()
# Run illumination process !
best = algo.optimise(eval_fn)

# Print results info
print(algo.summary())

# Plot the results
plots.default_plots_grid(logger)

print("All results are available in the '%s' pickle file." % logger.final_filename)