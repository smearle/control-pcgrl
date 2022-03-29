### Examples
A number of examples are provided in the `examples` directory [here](https://gitlab.com/leo.cazenille/qdpy/tree/master/examples).
Here are short descriptions of all example scripts:
 * `custom_eval_fn.py`: presents how one can specify a custom evaluation function and use MAP-Elites to illuminate it.
 * `rastrigin_short.py`: the simplest example. It describes how to illuminate the rastrigin function.
 * `rastrigin.py`: a more complex version of the previous script, with support for configuration files (by default, the "conf/rastrigin.yaml" configuration file).
 * `nslc_rastrigin.py`: illumination of the rastrigin function, but with NSLC instead of MAP-Elites.
 * `artificial_landscapes.py`: illumination of several artificial landscape functions (e.g. sphere, ackley, rosenbrock, etc) by MAP-Elites. This script was used to generate the data for the GECCO 2019 poster paper [Comparing reliability of grid-based quality-diversity algorithms using artificial landscapes](https://dl.acm.org/doi/pdf/10.1145/3319619.3321895).
 * `bipedal_walker/bipedal_walker.py`: illumination of the OpenAI Gym [BipedalWalker-v2](https://gym.openai.com/envs/BipedalWalker-v2/) environment. You need to install OpenAI gym, pybullet and Box2D to use it. A more detailed README can be found [here](examples/bipedal_walker).
 * `deap_map-elites_rastrigin_short.py`, `deap_map-elites_rastrigin.py` and `deap_nslc_rastrigin.py`: three versions of the rastrigin illumination example showcasing the QDpy-DEAP integration. You need to have the DEAP library installed to use them.
 * `deap_map-elites_SR.py`: illumination of a symbolic regression problem with MAP-Elites. It relies on the QDpy-DEAP integration.

All examples generate a pickle data file containing all results and the final container/grid. It can be accessed through the following code:
```python
import pickle
# You may want to import your own packages if the pickle file contains custom objects

with open("final.p", "rb") as f:
    data = pickle.load(f)
# ``data`` is now a dictionary containing all results, including the final container, all solutions, the algorithm parameters, etc.

grid = data['container']
print(grid.best)
print(grid.best.fitness)
print(grid.best.features)
```

They also generate two PDF plots:
 * `performanceGrid.pdf`: shows the fitness score of all individual in the final grid.
 * `activityGrid.pdf`: shows the 'activity' of the algorithm. We define the activity as the number of times a cell has been updated in the grid.

More examples will be added in the future.

