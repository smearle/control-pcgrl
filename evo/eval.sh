#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12

## We won't be asking for gpus, for now
##SBATCH --gres=gpu:1

#SBATCH --time=2:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=evalevopcgrl
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=zj2086@nyu.edu
#SBATCH --output=evalevopcgrl_%j.out

cd /scratch/zj2086/control-pcgrl

## Is this actually necessary?
source activate

## NOTE THIS ACTUALLY WORKS DONT LISTEN TO THE ERROR MESSAGE ???
conda activate evo-pcgrl

python evo/evolve.py -la 0