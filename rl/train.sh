#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=pcgrl_3D
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=zj2086@nyu.edu
#SBATCH --output=rl_runs/pcgrl_binary_ctrl_narrow_SeqNCA_3-scans_lr-1.0e-05_testSeq3D_%j.out

## cd /scratch/zj2086/control-pcgrl

## Is this actually necessary?
## ZJ: I don't think so? Calling conda activate <env_name> twice will throw an warning (but won't crash)
## source activate pcgrl

## NOTE THIS ACTUALLY WORKS DONT LISTEN TO THE ERROR MESSAGE ???
## conda activate pcgrl

start=$SECONDS
python rl/train_ctrl.py --load_args binary_ctrl_narrow_SeqNCA_3-scans_lr-1.0e-05_testSeq3D
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
