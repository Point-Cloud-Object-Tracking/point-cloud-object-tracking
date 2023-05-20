#!/bin/bash
# SBATCH --job-name=centerpoint-training
# SBATCH --partition=GPUNodes
# SBATCH --time=00:05:00
# SBATCH --mem=2G
# SBATCH --ntasks=1
# SBATCH --nodelist=node5
# SBATCH --gres=gpu:1,gpu_mem:8000
# SBATCH --gpus-per-task=1
# SBATCH --constraint=”fermi”
# SBATCH --nodelist=node2
# SBATCH --output=R-%x.%j-gpu-stat.out

######################################################################################
# NOTE: this is a SLURM script demo. Run it ala
#
#    sbatch /home/shared/slurm_demo.sh
#      Submitted batch job 9
#
#     $ squeue 
#             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
#                 9  GPUNodes somename      cef  R       0:04      1 node5
#
# and use scancel <jobid> if you want to cancel one of your jobs.
# 
# Read about the SBATCH parameters (above) on the 'net'.
#
# Enjoy
#  .c 
######################################################################################

# just some dummy commands, create you own script here..
gpustat -i 1