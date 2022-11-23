#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12

## We won't be asking for gpus, for now
##SBATCH --gres=gpu:1

#SBATCH --time=2:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=evalevopcg
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=se2161@nyu.edu
#SBATCH --output=evalevopcg_%j.out

## cd /scratch/se2161/evo-pcgrl

## Is this actually necessary?
## source activate

## NOTE THIS ACTUALLY WORKS DONT LISTEN TO THE ERROR MESSAGE ???
## conda activate pcgrl

python evolve.py -la 0
