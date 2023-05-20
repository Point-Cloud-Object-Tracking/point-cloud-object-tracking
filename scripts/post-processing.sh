#!/usr/bin/env bash

# set -x

Black='\033[0;30m'        # Black
Red='\033[0;31m'          # Red
Green='\033[0;32m'        # Green
Yellow='\033[0;33m'       # Yellow
Blue='\033[0;34m'         # Blue
Purple='\033[0;35m'       # Purple
Cyan='\033[0;36m'         # Cyan
White='\033[0;37m'        # White
Grey='\033[0;90m'         # Grey
NC='\033[0m'              # No Color

function cd_and_print_pwd() {
  cd "$1" || exit
  echo -e "${Grey}Moved to ${PWD}${NC}"
}

function check_exit_status() {
  eval "$@"
  if [[ ! $? -eq 0 ]]; then
    echo -e "${Red}Nonzero exit status${NC}"
    exit
  fi
}

CKPT=""
CONFIG_FILE=""
DET_NAME=""
BATCH_SIZE=16

# Loop through all the arguments
while (( "$#" )); do
  case "$1" in
    --ckpt)
      CKPT="$2"
      shift 2
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --id)
      DET_NAME="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      shift
      ;;
  esac
done

# Check if all variables are set
if [ -z "$CKPT" ] || [ -z "$CONFIG_FILE" ] || [ -z "$DET_NAME" ]; then
  echo "Error: model checkpoint, config file and detection name must be set"
  echo "Usage: $0 --ckpt CKPT --config CONFIG_FILE --id DET_NAME [--batch-size BATCH_SIZE]"
  exit 1
fi

# You can use these variables elsewhere in your script
echo "Checkpoint: $CKPT"
echo "Config: $CONFIG_FILE"
echo "ID: $DET_NAME"
echo "Batch Size: $BATCH_SIZE"

CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
EPOCH=$(basename "$CKPT" .pth | tr _ "\n" | tail -2 | xargs | tr " " "_")
if [[ ! $EPOCH =~ epoch_[0-9]+ ]]; then
  echo -e "${Red}Checkpoint filename has to be of the regex pattern '.*epoch_[0-9]+.*.pth'${NC}"
fi

DETECTION_OUTPUT="/home/cv08f23/point-cloud-object-tracking/OpenPCDet/output/${CONFIG_NAME}/default/eval/${EPOCH}/val/default/final_result/data/results_nusc.json"
echo -e "${Grey}Assuming ${DETECTION_OUTPUT} for object detection ouputs ${NC}"

# [1/6] Object Detections
echo -e "${Cyan}[1/6] Running Detections on model ${CKPT}${NC}"
cd_and_print_pwd ~/point-cloud-object-tracking/OpenPCDet/tools

check_exit_status python test.py --cfg_file "./cfgs/nuscenes_models/${CONFIG_NAME}.yaml" --batch_size "$BATCH_SIZE"  --ckpt "$CKPT"

# [2/6] SimpleTrack Motion Model
echo -e "${Cyan}[2/6] Running SimpleTrack motion model...${NC}"
cd_and_print_pwd ~/point-cloud-object-tracking/SimpleTrack/preprocessing/nuscenes_data

RAW_DATA_DIR=~/datasets/nuScenes/v1.0-trainval/

DATA_DIR_2HZ=~/datasets/simpletrack/preprocessing/nuscenes_data/2hz
DATA_DIR_20HZ=~/datasets/simpletrack/preprocessing/nuscenes_data/20hz

MODE=2hz # 20hz | 2hz
if [ "$MODE" = "20hz" ]; then
    DATA_DIR="$DATA_DIR_20HZ"
else
    DATA_DIR="$DATA_DIR_2HZ"
fi

echo -e "${Grey}Using ${MODE} mode${NC}"
echo -e "${Grey}Using ${DET_NAME} as identifier${NC}"

check_exit_status python detection.py \
    --raw_data_folder "$RAW_DATA_DIR" \
    --data_folder "$DATA_DIR" \
    --det_name "$DET_NAME" \
    --file_path "$DETECTION_OUTPUT" \
    --mode "$MODE" \
    --velo

# [3/6] SimpleTrack Tracking
echo -e "${Cyan}[3/6] Running SimpleTrack tracking...${NC}"
cd_and_print_pwd ~/point-cloud-object-tracking/SimpleTrack/tools/

RESULT_FOLDER="$HOME/datasets/simpletrack/tracking/nuscenes_data/$DET_NAME/"
echo -e "${Grey}Outputting tracking results to ${RESULT_FOLDER}${NC}"
CONFIG_PATH=~/point-cloud-object-tracking/SimpleTrack/configs/nu_configs/giou.yaml

check_exit_status python main_nuscenes.py \
    --det_name "$DET_NAME" \
    --config_path "$CONFIG_PATH" \
    --result_folder "$RESULT_FOLDER" \
    --data_folder "$DATA_DIR"

# [4/6] SimpleTrack Results
echo -e "${Cyan}[4/6] Generating result JSONs...${NC}"
cd_and_print_pwd ~/point-cloud-object-tracking/SimpleTrack/tools/

check_exit_status python nuscenes_result_creation.py \
    --result_folder "$RESULT_FOLDER" \
    --data_folder "$DATA_DIR"


# [5/6] Merge SimpleTrack Outputs
echo -e "${Cyan}[5/6] Mergin SimpleTrack result JSONs...${NC}"
cd_and_print_pwd "${RESULT_FOLDER}/debug/results"
check_exit_status merge_results.py

# [6/6] Tracking Evaluation
echo -e "${Cyan}[6/6] Evaluating tracking results...${NC}"

SET="val" # train | val | test
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

RESULTS_DATA_DIR="$HOME/datasets/simpletrack/tracking/$TRACKING_OUTPUT_PARENT_DIR/$DET_NAME/debug/results/merged_output.json"
echo -e "${Grey}Using $RESULTS_DATA_DIR for evaluation${NC}"

OUTPUT_DIR="$HOME/datasets/simpletrack/tracking/$TRACKING_OUTPUT_PARENT_DIR/$DET_NAME/debug/eval"
echo -e "${Grey}Outputting evaluations to ${OUTPUT_DIR}${NC}"

check_exit_status python ~/.conda/envs/pcot_3.9/lib/python3.9/site-packages/nuscenes/eval/tracking/evaluate.py \
       "$RESULTS_DATA_DIR" \
       --version "$VERSION" \
       --eval_set "$SET" \
       --dataroot "$RAW_DATA_FOLDER" \
       --output_dir "$OUTPUT_DIR"
