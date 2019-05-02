#!/bin/bash -l
#SBATCH --gres=gpu:v100:1
#SBATCH -t 960
#SBATCH -c 4
#SBATCH --mem=40Gb
module_name=$1
shift
python -m $module_name  $@
