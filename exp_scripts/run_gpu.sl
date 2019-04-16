#!/bin/bash -l
#SBATCH --gres=gpu
#SBATCH -t 960
module_name=$1
shift
python -m $module_name  $@
