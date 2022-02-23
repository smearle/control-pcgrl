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
#SBATCH --mail-user=zj2086@nyu.edu
#SBATCH --output=pcgrl_%j.out

cd /scratch/zj2086/gym-pcgrl

## Is this actually necessary?
source activate pcgrl

## NOTE THIS ACTUALLY WORKS DONT LISTEN TO THE ERROR MESSAGE ???
conda activate pcgrl

# start=$SECONDS
python train.py
# do
#     duration=$((( SECONDS - start ) / 60))
#     echo "Script returned error after $duration minutes"
#     if [ $minutes -lt 60 ]
#     then
#       echo "Too soon. Something is wrong. Terminating node."
#       exit
#     else
#       echo "Re-launching script."
#       start=$SECONDS
#     fi
# done
