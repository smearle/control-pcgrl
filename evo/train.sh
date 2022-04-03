#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48

## We won't be asking for gpus, for now
##SBATCH --gres=gpu:1

#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=evopcgrl
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=zj2086@nyu.edu
#SBATCH --output=evo_runs/evopcg_0_%j.out

cd /scratch/zj2086/control-pcgrl

## Is this actually necessary?
source activate

## NOTE THIS ACTUALLY WORKS DONT LISTEN TO THE ERROR MESSAGE ???
conda activate pcgrl

start=$SECONDS
while ! python evo/evolve.py -la 0
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
