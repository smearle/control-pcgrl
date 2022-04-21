#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10

## We won't be asking for gpus, for now
##SBATCH --gres=gpu:1

#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=evopcg
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=se2161@nyu.edu
#SBATCH --output=evo_runs/evopcg_62_%j.out

cd /scratch/se2161/control-pcgrl


# We try to activate the relevant conda environment below.

## Is this actually necessary?
source activate

## Is the error message here telling the truth?
conda activate pcgrl


start=$SECONDS
while ! python evo/evolve.py -la 62
do
    duration=$((( SECONDS - start ) / 60))
    echo "Script returned error after $duration minutes"
    if [ $duration -lt 60 ]
    then
      echo "Too soon. Something is wrong. Terminating node."
      exit 42
    else
      echo "Killing ray processes and re-launching script."
      ray stop
      pkill ray
      pkill -9 ray
      pkill python
      pkill -9 python
      start=$SECONDS
    fi
done
