python -m pip install -r requirements.txt
python -m pip install -e gym_pcgrl

# NOTE: only need this for evolving diverse generators, not RL.
python -m pip install -e submodules/qdpy


# Installing torch:

# for cpu
# conda install pytorch torchvision torchaudio -c pytorch

# for most GPUs (?)
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

# for 3090
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch


# GUI libraries for rendering (and controlling) controllable agents.
conda install -c conda-forge pygobject gtk3
