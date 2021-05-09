#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
##SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=pcgrl
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sam.earle@nyu.edu
#SBATCH --output=pcgrl_%j.out

cd /scratch/se2161/gym-pcgrl
#conda init bash
source activate

## NOTE THIS ACTUALLY WORKS DONT LISTEN TO THE ERROR MESSAGE ???
conda activate vanilla_pcgrl

# evo-pcgrl comparison
python train_controllable.py --problem "binary_ctrl" --representation "narrow" --conditionals "NONE" --evo_compare

#python train_controllable.py --problem "binary_ctrl" --conditionals "path-length" --representation "narrow"
#python train_controllable.py --problem "binary_ctrl" --conditionals "path-length" --representation "narrow" --alp_gmm
#python train_controllable.py --problem "binary_ctrl" --conditionals "regions" --representation "narrow"
#python train_controllable.py --problem "binary_ctrl" --conditionals "regions" "path-length" --representation "narrow"
#python train_controllable.py --problem "binary_ctrl" --conditionals "regions" "path-length" --representation "turtle"
#python train_controllable.py --problem "binary_ctrl" --conditionals "regions" "path-length" --representation "wide"

#python train_controllable.py --problem "zelda_ctrl" --conditionals "path-length" --representation "narrow"
#python train_controllable.py --problem "zelda_ctrl" --conditionals "nearest-enemy" --representation "narrow"
#python train_controllable.py --problem "zelda_ctrl" --conditionals "nearest-enemy" "path-length" --representation "narrow"
#python train_controllable.py --problem "zelda_ctrl" --conditionals "nearest-enemy" "path-length" --representation "narrow"
#python train_controllable.py --problem "zelda_ctrl" --conditionals "nearest-enemy" "path-length" --representation "turtle"
#python train_controllable.py --problem "zelda_ctrl" --conditionals "nearest-enemy" "path-length" --representation "wide"

#python train_controllable.py --problem "sokoban_ctrl" --conditionals "crate" --representation "narrow"
#python train_controllable.py --problem "sokoban_ctrl" --conditionals "sol-length" --representation "narrow"
#python train_controllable.py --problem "sokoban_ctrl" --conditionals "crate" "sol-length" --representation "narrow"
#python train_controllable.py --problem "sokoban_ctrl" --conditionals "crate" "sol-length" --representation "turtle"
#python train_controllable.py --problem "sokoban_ctrl" --conditionals "crate" "sol-length" --representation "wide"



## OLD FUCKING TRASH GARBAGE

### BINARY ###


## NARROW

#python train_controllable.py --problem "binary_ctrl" --representation "narrow"
#python train_controllable.py --problem "binary_ctrl" --conditionals "regions" --representation "narrow"
#python train_controllable.py --problem "binary_ctrl" --conditionals "path-length" --representation "narrow"

## TURTLE

#python train_controllable.py --problem "binary_ctrl" --representation "turtle"
#python train_controllable.py --problem "binary_ctrl" --conditionals "regions" --representation "turtle"
#python train_controllable.py --problem "binary_ctrl" --conditionals "path-length" --representation "turtle"
#python train_controllable.py --problem "binary_ctrl" --conditionals "ALL" --representation "turtle"


## WIDE

#python train_controllable.py --problem "binary_ctrl" --representation "wide"
#python train_controllable.py --problem "binary_ctrl" --conditionals "regions" --representation "wide" 
#python train_controllable.py --problem "binary_ctrl" --conditionals "path-length" --representation "wide"
#python train_controllable.py --problem "binary_ctrl" --conditionals "ALL" --representation "wide"

## WIDE - CA

#python train_controllable.py --problem "binary_ctrl" --representation "wide" --ca_action
#python train_controllable.py --problem "binary_ctrl" --conditionals "regions" --representation "wide" --ca_action
#python train_controllable.py --problem "binary_ctrl" --conditionals "path-length" --representation "wide" --ca_action
#python train_controllable.py --problem "binary_ctrl" --conditionals "ALL" --representation "wide" --ca_action



### ZELDA ###


## NARROW


#python train_controllable.py --problem "zelda_ctrl" --representation "narrow"
#python train_controllable.py --problem "zelda_ctrl" --conditionals "nearest-enemy" --representation "narrow"
#python train_controllable.py --problem "zelda_ctrl" --conditionals "path-length" --representation "narrow"

#python train_controllable.py --problem "zelda_ctrl" --conditionals "enemies" --representation "narrow"
#python train_controllable.py --problem "zelda_ctrl" --conditionals "enemies" "path-length" --representation "narrow"
#python train_controllable.py --problem "zelda_ctrl" --conditionals "ALL" --representation "narrow"

## TURTLE

#python train_controllable.py --problem "zelda_ctrl" --representation "turtle"
#python train_controllable.py --problem "zelda_ctrl" --conditionals "enemies" --representation "turtle"
#python train_controllable.py --problem "zelda_ctrl" --conditionals "path-length" --representation "turtle"
#python train_controllable.py --problem "zelda_ctrl" --conditionals "enemies" "path-length" --representation "turtle"
#python train_controllable.py --problem "zelda_ctrl" --conditionals "nearest-enemy" --representation "turtle"
#python train_controllable.py --problem "zelda_ctrl" --conditionals "player" "key" "door" "enemies" "regions" "nearest-enemy" "path-length" --representation "turtle"

## WIDE

#python train_controllable.py --problem "zelda_ctrl" --representation "wide"
#python train_controllable.py --problem "zelda_ctrl" --conditionals "player" "key" "door" "enemies" "regions" "nearest-enemy" "path-length" --representation "wide"

## WIDE - CA

#python train_controllable.py --problem "zelda_ctrl" --conditionals "player" "key" "door" "enemies" "regions" "nearest-enemy" "path-length" --representation "wide" --ca_action

### SOKOBAN

## NARROW


## TURTLE

#python train_controllable.py --problem "sokobangoal" --conditionals "crate" --representation "turtle"
#python train_controllable.py --problem "sokobangoal" --conditionals "sol-length" --representation "turtle"
#python train_controllable.py --problem "sokobangoal" --conditionals "crate" "sol-length" --representation "turtle"
#python train_controllable.py --problem "sokobangoal" --conditionals "player" "crate" "sol-length" --representation "turtle"

## WIDE

#python train_controllable.py --problem "sokobangoal" --conditionals "crate" --representation "wide"
#python train_controllable.py --problem "sokobangoal" --conditionals "sol-length" --representation "wide"
#python train_controllable.py --problem "sokobangoal" --conditionals "crate" "sol-length" --representation "wide"
#python train_controllable.py --problem "sokobangoal" --conditionals "player" "crate" "sol-length" --representation "wide"
 
## WIDE - CA

#python train_controllable.py --problem "sokobangoal" --conditionals "crate" --representation "wide" --ca_action
#python train_controllable.py --problem "sokobangoal" --conditionals "sol-length" --representation "wide" --ca_action
#python train_controllable.py --problem "sokobangoal" --conditionals "crate" "sol-length" --representation "wide" --ca_action
#python train_controllable.py --problem "sokobangoal" --conditionals "player" "crate" "sol-length" --representation "wide" --ca_action

