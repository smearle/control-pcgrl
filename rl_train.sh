#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48

## We won't be asking for gpus, for now
##SBATCH --gres=gpu:1

#SBATCH --time=120:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=pcgrl
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sam.earle@nyu.edu
#SBATCH --output=pcgrl_%j.out

cd /scratch/se2161/evo-pcgrl

## Is this actually necessary?
source activate vanilla_pcgrl

## NOTE THIS ACTUALLY WORKS DONT LISTEN TO THE ERROR MESSAGE ???
conda activate vanilla_pcgrl

start=$SECONDS
while ! python train_ctrl.py -la 0
do
    duration=$((( SECONDS - start ) / 60))
    echo "Script returned error after $duration minutes"
    if [ $minutes -lt 60 ]
    then
      echo "Too soon. Something is wrong. Terminating node."
      exit
    else
      echo "Re-launching script."
      start=$SECONDS
    fi
done

