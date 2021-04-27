#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12

## We won't be asking for gpus, for now
##SBATCH --gres=gpu:1

#SBATCH --time=120:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=evopcgrl
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sam.earle@nyu.edu
#SBATCH --output=evo-evopcgrl_%j.out

cd /scratch/se2161/evo-pcgrl

## Is this actually necessary?
source activate

## NOTE THIS ACTUALLY WORKS DONT LISTEN TO THE ERROR MESSAGE ???
conda activate evo-pcgrl

#python evolve.py --problem zelda -bcs nearest-enemy path-length -ng 1000000 -e "0" -m
#python evolve.py -e 'crate_sol_0' -bcs 'crate' 'sol-length' -p 'sokoban_ctrl' -m
#python evolve.py -e 'empty_sym_0' -bcs 'emptiness' 'symmetry' -p 'sokoban_ctrl' -m
#python evolve.py -e 'symmetry_empty_0' -bcs 'emptiness' 'symmetry' -p 'smb_ctrl' -m
python evolve.py -e 'jumps_empty_1' -bcs 'emptiness' 'jumps' -p 'smb_ctrl' -m
#python evolve.py -e 'empty_sym' -bcs 'emptiness' 'symmetry' -p ‘zeldaplay’ -m
