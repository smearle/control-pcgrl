python -m pip install -r requirements.txt
python -m pip install -e . submodules/qdpy

# for cpu
conda install pytorch torchvision torchaudio -c pytorch

# for torch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

# for 3090
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch