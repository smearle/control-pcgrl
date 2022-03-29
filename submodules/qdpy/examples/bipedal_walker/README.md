# Example with the OpenAI Gym BipedalWalker-v2 environment

This example showcases the illumination of the OpenAI Gym [BipedalWalker-v2](https://gym.openai.com/envs/BipedalWalker-v2/) environment with [MAP-Elites](https://arxiv.org/abs/1504.04909). Results obtained with this code can be found in the following GECCO2020 poster: [Exploring the BipedalWalker benchmark with MAP-Elites and Curiosity-driven A3C](TBD).

Several versions of this example are available in this directory:
 * `bipedal_walker.py`: baseline version. The neural network (MLP) is implemented directly using Numpy.
 * `bipedal_walker-pybrain.py`: version with the neural network using an implementation from the [PyBrain](http://pybrain.org/) library.
 * `bipedal_walker-pytorch.py`: neural network implemented with [PyTorch](https://pytorch.org/).
 * `bipedal_walker-tensorflow.py`: neural network implemented with [Tensorflow 2.0](https://www.tensorflow.org/) and tf.Keras.

## Launch locally
For the baseline version:
```bash
pip3 install gym box2d-py PyOpenGL setproctitle pybullet qdpy[all] cma
./bipedal_walker.py -c conf/test.yaml
```

The other versions can be launched in the same way, but you need to install the related packages before that:
```bash
pip3 install tensorflow pytorch pybrain
...
```

After the illumination process, it is possible to render simulations of the best-performing individual by using the following command:
```bash
./bipedal_walker.py -c conf/test.yaml --replayBestFrom PATH_TO_RESULTS_FILE.p
```


## Launch this example with [Docker](https://www.docker.com/)
```bash
./createDockerImage.sh          # Execute docker build
./runDockerExpe.sh test.yaml    # To launch an experiment with configuration file "conf/test.yaml"
```
All results will be saved in the "results/CONFIG_NAME" directory (e.g.: "results/test/" for config file "conf/test.yaml").

