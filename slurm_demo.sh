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
# cd /home/cv08f23/point-cloud-object-tracking/OpenPCDet/tools
# python train.py --cfg_file cfgs/nuscenes_models/cbgs_voxel_res3d_centerpoint_custom.yaml --ckpt /home/cv08f23/point-cloud-object-tracking/OpenPCDet/output/nuscenes_models/cbgs_voxel_res3d_centerpoint_custom/default/ckpt/latest_model.pth

# cd /home/cv08f23/point-cloud-object-tracking/OpenPCDet/tools
# python train.py --cfg_file cfgs/nuscenes_models/cbgs_pointpillar.yaml --ckpt /home/cv08f23/point-cloud-object-tracking/OpenPCDet/output/nuscenes_models/cbgs_pointpillar/default/ckpt/latest_model.pth
# cd ~/point-cloud-object-tracking/OpenPCDet/tools
# CKPT="$HOME/point-cloud-object-tracking/OpenPCDet/output/nuscenes_models/cbgs_pointpillar/default/ckpt/checkpoint_epoch_40.pth"
# CONFIG_FILE="./cfgs/nuscenes_models/cbgs_pointpillar.yaml"
# BATCH_SIZE=16
# python test.py --cfg_file "$CONFIG_FILE" --batch_size "$BATCH_SIZE"  --ckpt "$CKPT"
# post-processing.sh --ckpt /home/cv08f23/point-cloud-object-tracking/OpenPCDet/output/nuscenes_models/cbgs_pointpillar/default/ckpt/checkpoint_epoch_40.pth --config /home/cv08f23/point-cloud-object-tracking/OpenPCDet/output/nuscenes_models/cbgs_pointpillar/default/cbgs_pointpillar.yaml --id pp_0.16

# post-processing.sh --ckpt /home/cv08f23/point-cloud-object-tracking/OpenPCDet/output/nuscenes_models/cbgs_voxel_res3d_centerpoint_custom/default/ckpt/checkpoint_epoch_30.pth --config /home/cv08f23/point-cloud-object-tracking/OpenPCDet/output/nuscenes_models/cbgs_voxel_res3d_centerpoint_custom/default/cbgs_voxel_res3d_centerpoint_custom.yaml --id cp_0.06 --batch-size 32
post-processing.sh --ckpt /home/cv08f23/pre-trained-models/cbgs_voxel0075_centerpoint_nds_6648_epoch_20.pth \
                   --config /home/cv08f23/point-cloud-object-tracking/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml \
                   --id cp_0.075 \
                   --batch-size 32