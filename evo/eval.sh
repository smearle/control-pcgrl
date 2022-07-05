#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10

## We won't be asking for gpus, for now
##SBATCH --gres=gpu:1

#SBATCH --time=2:00:00
#SBATCH --mem=96GB
#SBATCH --job-name=evalevopcg
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=se2161@nyu.edu
#SBATCH --output=evalevopcg_%j.out

## cd /scratch/se2161/control-pcgrl


## We try to activate the relevant conda environment below.

## Is this actually necessary?
## ZJ: No my friend.
## source activate

## Is the error message here telling the truth?
## conda activate pcgrl


python evo/evolve.py -la 47