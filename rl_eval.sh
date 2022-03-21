#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

## We won't be asking for gpus, for now
##SBATCH --gres=gpu:1

#SBATCH --time=12:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=evalpcgrl
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sam.earle@nyu.edu
#SBATCH --output=evalpcgrl_%j.out

cd /scratch/se2161/evo-pcgrl || exit

source activate vanilla_pcgrl
## NOTE THIS ACTUALLY WORKS DONT LISTEN TO THE ERROR MESSAGE ???
conda activate vanilla_pcgrl

python evaluate_ctrl.py --load_arguments 0
