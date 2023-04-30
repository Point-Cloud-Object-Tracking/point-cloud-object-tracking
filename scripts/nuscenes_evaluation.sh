#!/usr/bin/env bash

SET=val # train | val | test
# CLASS=car # bicycle | bus | car | motorcycle | pedestrian | trailer | truck

if [ "$SET" = "test" ]; then
       VERSION="v1.0-test"
       RAW_DATA_FOLDER=~/datasets/nuScenes/v1.0-test/
       TRACKING_OUTPUT_PARENT_DIR="nuscenes_test"
else
       VERSION="v1.0-trainval"
       RAW_DATA_FOLDER=~/datasets/nuScenes/v1.0-trainval/
       TRACKING_OUTPUT_PARENT_DIR="nuscenes_data"
fi

RESULTS_DATA_DIR="~/datasets/simpletrack/tracking/$TRACKING_OUTPUT_PARENT_DIR/debug/results/merged_output.json"
echo "$RESULTS_DATA_DIR"
OUTPUT_DIR="~/datasets/simpletrack/tracking/$TRACKING_OUTPUT_PARENT_DIR/debug/eval"

python ~/.conda/envs/pcot_3.9/lib/python3.9/site-packages/nuscenes/eval/tracking/evaluate.py \
       "$RESULTS_DATA_DIR" \
       --version "$VERSION" \
       --eval_set "$SET" \
       --dataroot "$RAW_DATA_FOLDER" \
       --output_dir "$OUTPUT_DIR"
