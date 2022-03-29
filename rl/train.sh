#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12

#SBATCH --gres=gpu:1

#SBATCH --time=120:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=pcgrl_3D
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=zj2086@nyu.edu
#SBATCH --output=pcgrl-digger_%j.out

cd /scratch/zj2086/control-pcgrl

## Is this actually necessary?
source activate pcgrl-rllib

## NOTE THIS ACTUALLY WORKS DONT LISTEN TO THE ERROR MESSAGE ???
conda activate pcgrl-rllib

# start=$SECONDS
python rl/train_ctrl.py --load_args 0
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