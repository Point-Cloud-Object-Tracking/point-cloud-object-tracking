# Point Cloud Object Tracking

> This repository contains the code used for our project on point cloud object tracking. The project was done as part of the course "Computer Vision spring 2023" at Aarhus University.

***

## How to install

### Clone the repository

The repository can be cloned using the following command:

```bash
git clone https://github.com/Point-Cloud-Object-Tracking/point-cloud-object-tracking.git --recursive
```

`SimpleTrack` and `OpenPCDet` are included as submodules. To update the submodules, run the following command:

```bash
git submodule update --init --recursive
```

### Dependencies

The code is written in Python 3.9.0 and uses `conda` to manage dependencies. To install `conda`, follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

The dependencies are listed in the `environment.yaml` file.

```bash
conda env export --from-history
```

```yaml
name: pcot_3.9
channels:
  - defaults
  - conda-forge
dependencies:
  - python==3.9
  - torchaudio
  - torchvision
  - pytorch
  - pytorch-cuda=11.8
  - tensorboardx
  - sharedarray
  - cudatoolkit
prefix: /home/cv08f23/.conda/envs/pcot_3.9
```

To create a new environment with the dependencies, run the following command:

```bash
conda env create -n <env_name> -f environment.yaml
```

Both `SimpleTrack` and `OpenPCDet` comes with their own dependencies. These are listed in the `requirements.txt` files in the respective folders. To install the dependencies, run the following commands:

```bash
python -m pip install -r ./SimpleTrack/requirements.txt
```

`OpenPCDet` also depends on [traveller59/spconv: Spatial Sparse Convolution Library](https://github.com/traveller59/spconv). This library can be GPU accelerated with Nvidia CUDA, but needs to be explicitly installed with 
that option to do so. Our develop environment has CUDA 12.0 installed, so we need to install `spconv` with CUDA 12.0 support. This can be done by running the following commands:

```bash
python -m pip install spconv-cu120
```
 
If you have a different version of CUDA installed, you can find the version of `spconv` that matches your CUDA version [here](https://github.com/traveller59/spconv#install)


```bash
pushd OpenPCDet
python -m pip install -r ./requirements.txt

# Install this `pcdet` library and its dependent libraries by running the following command:
python setup.py develop
```

## Datasets

The datasets used in the project are listed below. The datasets are not included in the repository and must be downloaded separately.

### KITTI

The KITTI dataset can be downloaded from [here](http://www.cvlibs.net/datasets/kitti/eval_tracking.php).

We extract the data from the `training` and `testing` folders and place them in the following structure:

```
~/datasets/kitti/
├── devkit
│   ├── cpp
│   ├── mapping
│   └── matlab
├── gt_database
├── ImageSets
├── testing
│   ├── calib
│   ├── image_2
│   ├── tmp
│   └── velodyne
└── training
    ├── calib
    ├── image_2
    ├── label_2
    └── velodyne
```

### nuScenes

The nuScenes dataset can be downloaded from [here](https://www.nuscenes.org/download).

We extract the data from the `v1.0-trainval` and `v1.0-test` folders and place them in the following structure:

```
~/datasets/nuScenes/
├── detection
│   └── cp
│       └── dets
├── v1.0-test
│   ├── gt_database_10sweeps_withvelo
│   ├── maps
│   ├── samples
│   │   ├── CAM_BACK
│   │   ├── CAM_BACK_LEFT
│   │   ├── CAM_BACK_RIGHT
│   │   ├── CAM_FRONT
│   │   ├── CAM_FRONT_LEFT
│   │   ├── CAM_FRONT_RIGHT
│   │   ├── LIDAR_TOP
│   │   ├── RADAR_BACK_LEFT
│   │   ├── RADAR_BACK_RIGHT
│   │   ├── RADAR_FRONT
│   │   ├── RADAR_FRONT_LEFT
│   │   └── RADAR_FRONT_RIGHT
│   ├── sweeps
│   │   ├── CAM_BACK
│   │   ├── CAM_BACK_LEFT
│   │   ├── CAM_BACK_RIGHT
│   │   ├── CAM_FRONT
│   │   ├── CAM_FRONT_LEFT
│   │   ├── CAM_FRONT_RIGHT
│   │   ├── LIDAR_TOP
│   │   ├── RADAR_BACK_LEFT
│   │   ├── RADAR_BACK_RIGHT
│   │   ├── RADAR_FRONT
│   │   ├── RADAR_FRONT_LEFT
│   │   └── RADAR_FRONT_RIGHT
│   └── v1.0-test
└── v1.0-trainval
    ├── gt_database_10sweeps_withvelo
    ├── maps
    ├── samples
    │   ├── CAM_BACK
    │   ├── CAM_BACK_LEFT
    │   ├── CAM_BACK_RIGHT
    │   ├── CAM_FRONT
    │   ├── CAM_FRONT_LEFT
    │   ├── CAM_FRONT_RIGHT
    │   ├── LIDAR_TOP
    │   ├── RADAR_BACK_LEFT
    │   ├── RADAR_BACK_RIGHT
    │   ├── RADAR_FRONT
    │   ├── RADAR_FRONT_LEFT
    │   └── RADAR_FRONT_RIGHT
    ├── sweeps
    │   ├── CAM_BACK
    │   ├── CAM_BACK_LEFT
    │   ├── CAM_BACK_RIGHT
    │   ├── CAM_FRONT
    │   ├── CAM_FRONT_LEFT
    │   ├── CAM_FRONT_RIGHT
    │   ├── LIDAR_TOP
    │   ├── RADAR_BACK_LEFT
    │   ├── RADAR_BACK_RIGHT
    │   ├── RADAR_FRONT
    │   ├── RADAR_FRONT_LEFT
    │   └── RADAR_FRONT_RIGHT
    └── v1.0-trainval
```

### Creating symbolic links to the datasets

To make it easier to work with the datasets, we create symbolic links to the datasets in the `OpenPCDet` folderse. This is done by running the following commands:

```bash
ln -sf ~/datasets/kitti $PWD/OpenPCDet/data/kitti
ln -sf ~/datasets/nuScenes $PWD/OpenPCDet/data/nuScenes
```

**NOTE** if you have placed the datasets in a different location, you will need to change the paths in the above commands accordingly.

## Pretrained models

`OpenPCDet` provides pretrained models for the nuScenes dataset [here](https://github.com/open-mmlab/OpenPCDet#nuscenes-3d-object-detection-baselines) and KITTI [here](https://github.com/open-mmlab/OpenPCDet#kitti-3d-object-detection-baselines)


## How to train a model from scratch

Here is how to train the pointpillar model on the nuScenes dataset.

```bash
cd OpenPCDet/tools
python train.py \
    --cfg_file cfgs/nuscenes_models/cbgs_pointpillar.yaml \
    --ckpt $PWD/OpenPCDet/output/nuscenes_models/cbgs_pointpillar/default/ckpt/latest_model.pth
```

## How to evaluate a model

Here is how to evaluate one of the 3D detection models .e.g. PointPillars on the nuScenes dataset.
It will first use the model to infer 3D bounding boxes on the test set.
Then it will use `SimpleTrack` to track the objects in the test set.
Finally, it will evaluate the tracking results using the official nuScenes evaluation script, 
from `nuScenes-devkit/python-sdk/nuscenes/eval/detection/evaluate.py`.


```bash
./scripts/post_processing.sh \
    --ckpt <path_to_ckpt> \ # e.g. OpenPCDet/output/nuscenes_models/cbgs_pointpillar/default/ckpt/latest_model.pth
    --config <path_to_config> \ # e.g. OpenPCDet/output/nuscenes_models/cbgs_pointpillar/default/config.yaml
    --id <experiment_id> \ # e.g. pointpillar
    --batch-size <batch_size> # e.g. 16
```

The different metrics are saved in the `results` folder, and will also be printed to stdout.

If you experience any issues with paths not being found, you can open the `post_processing.sh` script and change the paths, to match the directory where you have put your copy of the datasets.
