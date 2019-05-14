#!/bin/bash -l
#SBATCH --gres=gpu
#SBATCH -t 960
#SBATCH -c 6
#SBACTH --qos=unkillable
#SBATCH --mem=32Gb
module_name=$1
shift
python -m $module_name  $@
