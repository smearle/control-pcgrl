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
#SBATCH --mail-user=zj2086@nyu.edu
#SBATCH --output=evalpcgrl_%j.out

## cd /scratch/zj2086/control-pcgrl || exit

## source activate pcgrl
## NOTE THIS ACTUALLY WORKS DONT LISTEN TO THE ERROR MESSAGE ???
## conda activate pcgrl

python rl/evaluate_ctrl.py --load_args 0
