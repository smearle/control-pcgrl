python -m pip install -r requirements.txt
python -m pip install -e .
# python -m pip install -e gym_pcgrl

# NOTE: only need this for evolving diverse generators, not RL.
python -m pip install -e submodules/qdpy


####### Installing torch: #######

# for CPU
# conda install pytorch torchvision torchaudio -c pytorch

# RUN: nvcc --version to check your CUDA version

# for most GPUs (?) 
# conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

# for cuda version 11.6 on linux
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

# for 3090
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# # M1 mac users, if want to use GPU (MPS acceleration)):
# conda install pytorch torchvision torchaudio -c pytorch-nightly

#################################

# GUI libraries for rendering (and controlling) controllable agents.
conda install -c conda-forge pygobject gtk3

# Installing minerl:
# python -m pip install --upgrade minerl


# Installing hydra:
python -m pip install --upgrade hydra-core                 
# store hyperparam sweeps in separate files (in "experiments") will only work with hydra 1.2.0, but not 100% sure.
python -m pip install --upgrade hydra-submitit-launcher