#!/bin/bash

# Parameters
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=10
#SBATCH --error=/scratch/rd2893/control-pcgrl/control_pcgrl/configs/multirun/2022-11-29/15-04-08/.submitit/%A_%a/%A_%a_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=train_ctrl
#SBATCH --mem=30GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/scratch/rd2893/control-pcgrl/control_pcgrl/configs/multirun/2022-11-29/15-04-08/.submitit/%A_%a/%A_%a_0_log.out
#SBATCH --signal=USR2@120
#SBATCH --time=1440
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /scratch/rd2893/control-pcgrl/control_pcgrl/configs/multirun/2022-11-29/15-04-08/.submitit/%A_%a/%A_%a_%t_log.out --error /scratch/rd2893/control-pcgrl/control_pcgrl/configs/multirun/2022-11-29/15-04-08/.submitit/%A_%a/%A_%a_%t_log.err /scratch/rd2893/miniconda3/envs/pcgrl/bin/python -u -m submitit.core._submit /scratch/rd2893/control-pcgrl/control_pcgrl/configs/multirun/2022-11-29/15-04-08/.submitit/%j
