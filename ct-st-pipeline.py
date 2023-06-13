# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: pcot
#     language: python
#     name: python3
# ---

# # End-to-end object detection and tracking using Point Pillars and Simple Track

# ### Using the PointPillars pretrained model on the KITTI test dataset:

# +
# run PointPillars on KITTI dataset

# %cd ~/point-cloud-object-tracking/OpenPCDet/tools
# !python demo.py \
#     --cfg_file cfgs/kitti_models/pointpillar.yaml \
#     --ckpt ~/pre-trained-models/pointpillars-kitty.pth \
#     --data_path ~/datasets/kitti/testing/velodyne  \
# 
# -

# #### PointPillar KITTI Detections

# + language="bash"
# cd ~/point-cloud-object-tracking/OpenPCDet/tools
# CKPT="$HOME/pre-trained-models/pointpillars-kitty.pth"
# CONFIG_FILE="./cfgs/kitti_models/pointpillar.yaml"
# BATCH_SIZE=32
# python test.py --cfg_file "$CONFIG_FILE" --batch_size "$BATCH_SIZE"  --ckpt "$CKPT" --cuda_idx 1
# -

# ### Using the PointPillars-multihead pretrained model on the nuScenes test dataset:

# +
# run PointPillars-mutlihead on nuScenes dataset

# %cd ~/point-cloud-object-tracking/SimpleTrack/tools
# !python demo.py \
#     --name demo \
#     --det_name cp \
#     --obj_type vehicle \
#     --config_path ../configs/waymo_configs/vc_kf_giou.yaml \
#     --data_folder ./demo_data \
#     --visualize


# +
# Visualize samples from object detection
# %cd

from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


folder_path = "/home/cv08f23/results/kitti/point-pillars"
image_list = os.listdir(folder_path)[:2] # Get the first 5 images from the folder
for image_name in image_list:
    image_path = os.path.join(folder_path, image_name)
    image = Image.open(image_path)
    img_array = np.array(image)
    # display the image inline in the notebook
    plt.imshow(img_array)
    plt.axis("off")
    plt.show()

# -

# ### Using CenterPoint pretrained model on the nuScenes test dataset:

# + language="bash"
# cd ~/point-cloud-object-tracking/OpenPCDet/tools
# CKPT="$HOME/pre-trained-models/centerpoint-voxel01-nuscenes.pth"
# CONFIG_FILE="./cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint.yaml"
# CONFIG_FILE="./cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint_test.yaml"
# BATCH_SIZE=32
# python test.py --cfg_file "$CONFIG_FILE" --batch_size "$BATCH_SIZE"  --ckpt "$CKPT"
#
# -

# ### Using Trained PointPillars on nuScenes Validation Datasest

# + language="bash"
# cd ~/point-cloud-object-tracking/OpenPCDet/tools
# CKPT="$HOME/point-cloud-object-tracking/OpenPCDet/output/nuscenes_models/cbgs_pointpillar/default/ckpt/checkpoint_epoch_40.pth"
# CONFIG_FILE="./cfgs/nuscenes_models/cbgs_pointpillar.yaml"
# BATCH_SIZE=16
# python test.py --cfg_file "$CONFIG_FILE" --batch_size "$BATCH_SIZE"  --ckpt "$CKPT"
# -

# ## SimpleTrack Preprocessing

# + language="bash"
# cd ~/point-cloud-object-tracking/SimpleTrack/preprocessing/nuscenes_data
#
# # RAW_DATA_FOLDER=~/datasets/nuScenes/v1.0-trainval/
# RAW_DATA_FOLDER=~/datasets/nuScenes/v1.0-test/
#
# # DATA_DIR_2HZ=~/datasets/simpletrack/preprocessing/nuscenes_data/2hz
# # DATA_DIR_20HZ=~/datasets/simpletrack/preprocessing/nuscenes_data/20hz
# DATA_DIR_2HZ=~/datasets/simpletrack/preprocessing/nuscenes_test/2hz
# DATA_DIR_20HZ=~/datasets/simpletrack/preprocessing/nuscenes_test/20hz
#
# bash nuscenes_preprocess.sh "$RAW_DATA_FOLDER" "$DATA_DIR_2HZ" "$DATA_DIR_20HZ" "v1.0-test"

# +
import numpy as np
from pprint import pprint as pp
# data_path = "/home/cv08f23/results/kitti/point-pillars/test.npz"
data_path = "/home/cv08f23/datasets/simpletrack/preprocessing/nuscenes_test/2hz/pc/raw_pc/scene-0078.npz"
data = np.load(data_path, allow_pickle=True)
# Set the print options
np.set_printoptions(precision=2, suppress=True)
# for key in data:
#     print(f"Key: {key}")
#     print("Array:")
#     print(data[key])
#     print()

print(len(data.keys()))

# 3D visualization

# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': '3d'})
# ax.scatter(data['0'][:, 0], data['0'][:, 1], data['0'][:, 2], s=0.1)
# # equal axis
# ax.set_aspect("equal", adjustable="box")
# -

# ## SimpleTrack Motion Model

# + language="bash"
#
# cd ~/point-cloud-object-tracking/SimpleTrack/preprocessing/nuscenes_data
#
# RAW_DATA_DIR=~/datasets/nuScenes/v1.0-trainval/
# # RAW_DATA_DIR=~/datasets/nuScenes/v1.0-test/
#
# DATA_DIR_2HZ=~/datasets/simpletrack/preprocessing/nuscenes_data/2hz
# DATA_DIR_20HZ=~/datasets/simpletrack/preprocessing/nuscenes_data/20hz
# # DATA_DIR_2HZ=~/datasets/simpletrack/preprocessing/nuscenes_test/2hz
# # DATA_DIR_20HZ=~/datasets/simpletrack/preprocessing/nuscenes_test/20hz
#
# MODE=2hz # 20hz | 2hz
# if [ "$mode" = "20hz" ]; then
#     DATA_DIR="$DATA_DIR_20HZ"
# else
#     DATA_DIR="$DATA_DIR_2HZ"
# fi
#
# DET_NAME="cp_0.06"
# # FILE_PATH=~/point-cloud-object-tracking/OpenPCDet/output/cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint/default/eval/epoch_01/val/default/final_result/data/results_nusc.json
# # FILE_PATH=~/point-cloud-object-tracking/OpenPCDet/output/cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint/default/eval/epoch_01/test/default/final_result/data/results_nusc.json
# FILE_PATH=~/point-cloud-object-tracking/OpenPCDet/output/nuscenes_models/cbgs_voxel_res3d_centerpoint_custom/default/eval/eval_with_train/epoch_30/val/final_result/data/results_nusc.json
# echo "MODE: $MODE"
#
# python detection.py \
#     --raw_data_folder "$RAW_DATA_DIR" \
#     --data_folder "$DATA_DIR" \
#     --det_name "$DET_NAME" \
#     --file_path "$FILE_PATH" \
#     --mode "$MODE" \
#     --velo

# +
import numpy as np
from pprint import pprint as pp

data_path = "/home/cv08f23/datasets/simpletrack/preprocessing/nuscenes_test/2hz/detection/cp/dets/scene-0077.npz"
data = np.load(data_path, allow_pickle=True)

# Set the print options
np.set_printoptions(precision=2, suppress=True)

for key in data:
    print(f"Key: {key}")
    # print("Array:")
    # pp(data[key])
    # print()

# [x, y, z, w, l, h, qx, qy, qz, qw, score]
frame_idx = 1
print()
print(np.array(data['bboxes'][0])[frame_idx])
print(np.array(data['types'][0])[frame_idx])
print(np.array(data['velos'][0])[frame_idx])
# -

# ## SimpleTrack Tracking

# + language="bash"
#
# cd ~/point-cloud-object-tracking/SimpleTrack/tools/
#
# DET_NAME="cp_0.06"
#
# RESULT_FOLDER="$HOME/datasets/simpletrack/tracking/nuscenes_data/$DET_NAME/"
# # RESULT_FOLDER=~/datasets/simpletrack/tracking/nuscenes_test/
#
# CONFIG_PATH=~/point-cloud-object-tracking/SimpleTrack/configs/nu_configs/giou.yaml
#
# DATA_DIR_2HZ=~/datasets/simpletrack/preprocessing/nuscenes_data/2hz
# DATA_DIR_20HZ=~/datasets/simpletrack/preprocessing/nuscenes_data/20hz
# # DATA_DIR_2HZ=~/datasets/simpletrack/preprocessing/nuscenes_test/2hz
# # DATA_DIR_20HZ=~/datasets/simpletrack/preprocessing/nuscenes_test/20hz
#
# MODE=2hz # 20hz | 2hz
# if [ "$mode" = "20hz" ]; then
#     DATA_DIR="$DATA_DIR_20HZ"
# else
#     DATA_DIR="$DATA_DIR_2HZ"
# fi
#
# python main_nuscenes.py \
#     --det_name "$DET_NAME" \
#     --config_path "$CONFIG_PATH" \
#     --result_folder "$RESULT_FOLDER" \
#     --data_folder "$DATA_DIR"

# +
import numpy as np

data_path = "/home/cv08f23/datasets/simpletrack/tracking/nuscenes_test/debug/summary/truck/scene-0077.npz"
data = np.load(data_path, allow_pickle=True)

# Set the print options
np.set_printoptions(precision=2, suppress=True)

for key in data:
    print(f"Key: {key}")
    # print("Array:")
    # pp(data[key])
    # print()

frame_idx = 1
tracklet_idx = 1

# <scene>_<id>
# [<x>, <y>, <z>, <o>, <l>, <w>, <h>, <score>]
# <state>_<recent-state>_<time-since-last-update>
# <type>

print()
print(np.array(data['ids'][frame_idx])[tracklet_idx])
print(np.array(data['bboxes'][frame_idx])[tracklet_idx])
print(np.array(data['states'][frame_idx])[tracklet_idx])
print(np.array(data['types'][frame_idx])[tracklet_idx])
# -

# ## SimpleTrack Results

# + language="bash"
#
# cd ~/point-cloud-object-tracking/SimpleTrack/tools/
#
# DET_NAME="cp_0.06"
#
# RESULT_FOLDER="$HOME/datasets/simpletrack/tracking/nuscenes_data/$DET_NAME/"
# # RESULT_FOLDER=~/datasets/simpletrack/tracking/nuscenes_test/
#
# DATA_DIR_2HZ=~/datasets/simpletrack/preprocessing/nuscenes_data/2hz
# DATA_DIR_20HZ=~/datasets/simpletrack/preprocessing/nuscenes_data/20hz
# # DATA_DIR_2HZ=~/datasets/simpletrack/preprocessing/nuscenes_test/2hz
# # DATA_DIR_20HZ=~/datasets/simpletrack/preprocessing/nuscenes_test/20hz
#
# MODE=2hz # 20hz | 2hz
# if [ "$mode" = "20hz" ]; then
#     DATA_DIR="$DATA_DIR_20HZ"
# else
#     DATA_DIR="$DATA_DIR_2HZ"
# fi
#
# python nuscenes_result_creation.py \
#     --result_folder "$RESULT_FOLDER" \
#     --data_folder "$DATA_DIR"

# + language="bash"
#
# SET=val # train | val | test
# # CLASS=car # bicycle | bus | car | motorcycle | pedestrian | trailer | truck
#
# if [ "$SET" = "test" ]; then
#        VERSION="v1.0-test"
#        RAW_DATA_FOLDER=~/datasets/nuScenes/v1.0-test/
#        TRACKING_OUTPUT_PARENT_DIR="nuscenes_test"
# else
#        VERSION="v1.0-trainval"
#        RAW_DATA_FOLDER=~/datasets/nuScenes/v1.0-trainval/
#        TRACKING_OUTPUT_PARENT_DIR="nuscenes_data"
# fi
#
# DET_NAME="cp_0.06"
#
# RESULTS_DATA_DIR="~/datasets/simpletrack/tracking/$TRACKING_OUTPUT_PARENT_DIR/$DET_NAME/debug/results/merged_output.json"
# echo "$RESULTS_DATA_DIR"
# OUTPUT_DIR="~/datasets/simpletrack/tracking/$TRACKING_OUTPUT_PARENT_DIR/$DET_NAME/debug/eval"
#
# python ~/.conda/envs/pcot_3.9/lib/python3.9/site-packages/nuscenes/eval/tracking/evaluate.py \
#        "$RESULTS_DATA_DIR" \
#        --version "$VERSION" \
#        --eval_set "$SET" \
#        --dataroot "$RAW_DATA_FOLDER" \
#        --output_dir "$OUTPUT_DIR"
#
