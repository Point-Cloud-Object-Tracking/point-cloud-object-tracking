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

# +
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import os
import re
import seaborn as sns
import torch
import glob
import yaml
from OpenPCDet.pcdet.models import build_network, load_data_to_gpu
from OpenPCDet.pcdet.datasets import DatasetTemplate
from OpenPCDet.pcdet.utils import common_utils

# import open3d
from OpenPCDet.tools.visual_utils import open3d_vis_utils as V
OPEN3D_FLAG = True

# %matplotlib widget

# +
# load KITTI test dataset
class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


data_path = 'datasets/kitti/testing/velodyne'
cfg_path = '/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml'

with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

logger = common_utils.create_logger()

demo_dataset = DemoDataset(
    dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    root_path=data_path, ext='.bin', logger=logger
)

logger.info(f'Total number of samples: \t{len(demo_dataset)}')

# +
model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
model.load_params_from_file(filename='pre-trained-models/pointpillars.pth', logger=logger, to_cpu=True)
model.cuda()
model.eval()
with torch.no_grad():
    for idx, data_dict in enumerate(demo_dataset):
        logger.info(f'Visualized sample index: \t{idx + 1}')
        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        pred_dicts, _ = model.forward(data_dict)

        V.draw_scenes(
            points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
        )

        # if not OPEN3D_FLAG:
        #     mlab.show(stop=True)

logger.info('Demo done.')
# -

!python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ~/pre-trained-models/pointpillars-kitty.pth --data_path ~/datasets/kitti/training/velodyne/000008.bin
