import json
import os
import re
import warnings
from typing import Any, Dict

import sys
sys.path.append("/home/vblot/follicle-assessment/")

import cv2
import numpy as np
import patchify
import torch
from tqdm import tqdm
from mmcv import Config
from mmdet.apis import single_gpu_test
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils import build_dp, replace_cfg_vals, update_data_root
from mmcv.runner import load_checkpoint, wrap_fp16_model

from projects import *
from utils.depths_utils import compute_prediction_depths
from utils.patch_utils import is_not_white


warnings.filterwarnings("ignore")

CONFIG_FILE = "/home/vblot/Co-DETR/work_dirs/co_deformable_detr_r50_1x_coco/20240415_075344/co_deformable_detr_r50_1x_coco.py"
CHECKPOINT_PATH = "/home/vblot/Co-DETR/work_dirs/co_deformable_detr_r50_1x_coco/20240415_075344/epoch_8.pth"
PATCH_SIZE = 1000
OVARY_PATH = "/mnt/folcon/01_ovary_cuts/ovaries_images"
CLASS_NAME_VALUES = {0: "PMF", 1: "Primary", 2: "Secondary"}
BBOXES_SIZE_PARAMS = {
    "PMF": {"width": 200, "height": 200},
    "Primary": {"width": 350, "height": 350},
    "Secondary": {"width": 450, "height": 450}
}
SAVE_PREDICTIONS_PATH = "results.json"
DATA_SPLIT_PATH = "/mnt/folcon/02_model_input/data_split.json"
IMG_METAS_BASE = {
        "img_prefix": "temp/",
        "img_info": {
            "width": 1000,
            "height": 1000
        },
        "batch_input_shape": (1000, 1000),
        'img_shape': (800, 800, 3),
        'pad_shape': (800, 800, 3),
        # 'scale_factor': np.array([0.8, 0.8, 0.8, 0.8], dtype=np.float32),
        'ori_shape': (1000, 1000, 3),
        'flip': False,
        'flip_direction': None,
        'img_norm_cfg': {
            'mean': np.array([123.675, 116.28 , 103.53 ], dtype=np.float32),
            'std': np.array([58.395, 57.12 , 57.375], dtype=np.float32),
            'to_rgb': True
        }
}
TEST_DATALOADER_DEFAULT_ARGS = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
BASE_ANNOT = annots = {
    "info": {
        "description": "temp",
        "url": "temp",
        "version": "temp",
        "year": 2021,
        "contributor": "temp",
        "date_created": "temp"
    },
    "licenses": [
        {
            "url": "temp",
            "id": 0,
            "name": "temp"
        }
    ],
    "categories": [
        {
            "supercategory": "temp",
            "id": 0,
            "name": "PMF"
        },
        {
            "supercategory": "temp",
            "id": 1,
            "name": "Primary"
        },
        {
            "supercategory": "temp",
            "id": 2,
            "name": "Secondary"
        }
    ],
    "images": [
        {
            "height": 1000,
            "width": 1000,
            "id": 0
        }
    ],
    "annotations": []
}

def predict_detr(
            config_path: str,
            checkpoint_path: str,
            patch_size: int,
            ovary_path: str,
            data_split: Dict,
            class_name_values: Dict[int, str],
            bboxes_size_params
    ) -> Dict[str, Dict[str, Any]]:

    cfg = Config.fromfile(config_path)
    cfg = replace_cfg_vals(cfg)
    cfg.data.test.test_mode = True

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')).to("cuda")
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cuda')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model = build_dp(model, device="cuda", device_ids=[0])
    test_loader_cfg = {
        **TEST_DATALOADER_DEFAULT_ARGS,
        **cfg.data.get('test_dataloader', {})
    }
    predictions = {}
    # for ovary_id in tqdm(os.listdir(os.path.join(os.getcwd(), ovary_path))):
    for ovary_id in tqdm(data_split["test"]):
        predictions[ovary_id] = {}
        for cut_name in os.listdir(os.path.join(ovary_path, ovary_id)):
            roi_name = re.findall(r"roi\d+", cut_name)[0]
            predictions[ovary_id][roi_name] = {"bboxes": [], "scores": [], "classes": []}
            cut = cv2.imread(os.path.join(ovary_path, ovary_id, cut_name))
            patches = patchify.patchify(cut, (patch_size, patch_size, 3), step=patch_size)
            for ix in range(patches.shape[0]):
                for iy in range(patches.shape[1]):
                    if is_not_white(patches[ix, iy, 0, :, :, :]):
                        patch_row = ix
                        patch_col = iy
                        patch = patches[ix, iy, 0, :, :, :]
                        patch = cv2.resize(patch, (800, 800))
                        # Save patch in temp folder
                        cv2.imwrite(f"temp/{ix}_{iy}.png", patch)
                        # create annot file with coco format for this patch
                        annots = BASE_ANNOT.copy()
                        annots["images"][0]["file_name"] = f"{ix}_{iy}.png"
                        with open("temp/annot.json", "w") as f:
                            json.dump(annots, f)

                        temp_config_data = cfg.data.test
                        temp_config_data["ann_file"] = "temp/annot.json"
                        temp_config_data["img_prefix"] = "temp/"
                        dataset = build_dataset(temp_config_data)
                        data_loader = build_dataloader(dataset, **test_loader_cfg)
                        with torch.no_grad():
                            bbox_result = single_gpu_test(model, data_loader)
                        os.remove(f"temp/{ix}_{iy}.png")
                        os.remove("temp/annot.json")
                        scale_normalizatation_x = patch_size / cfg["test_pipeline"][1].img_scale[0]
                        scale_normalizatation_y = patch_size / cfg["test_pipeline"][1].img_scale[1]
                        concat_preds = np.concatenate([bbox_result[0][0], bbox_result[0][1], bbox_result[0][2]], axis=0)
                        # concat_preds[:, 0] *= scale_normalizatation_x
                        # concat_preds[:, 1] *= scale_normalizatation_y
                        # concat_preds[:, 2] *= scale_normalizatation_x
                        # concat_preds[:, 3] *= scale_normalizatation_y
                        # print(max(concat_preds[:, 0]), max(concat_preds[:, 1]), max(concat_preds[:, 2]), max(concat_preds[:, 3]))

                        pred_classes = np.array(
                            [0] * len(bbox_result[0][0]) + [1] * len(bbox_result[0][1]) + [2] * len(bbox_result[0][2])
                        )
                        pred_boxes = concat_preds[:, :4]
                        pred_confs = concat_preds[:, 4]
                        if len(pred_boxes) > 0:
                            for i in range(len(pred_boxes)):
                                x1 = pred_boxes[i][0] + patch_col * patch_size
                                y1 = pred_boxes[i][1] + patch_row * patch_size
                                x2 = pred_boxes[i][2] + patch_col * patch_size
                                y2 = pred_boxes[i][3] + patch_row * patch_size

                                class_label = pred_classes[i]
                                class_name = class_name_values[int(class_label)]
                                new_bbox_width = bboxes_size_params[class_name]["width"]
                                new_bbox_height = bboxes_size_params[class_name]["height"]
                                new_bbox = np.zeros(4)
                                new_bbox[0] = x1
                                new_bbox[1] = y1
                                new_bbox[2] = x1 + new_bbox_width
                                new_bbox[3] = y1 + new_bbox_height
                                predictions[ovary_id][roi_name]["bboxes"].append(new_bbox.tolist())
                                predictions[ovary_id][roi_name]["scores"].append(float(pred_confs[i]))
                                predictions[ovary_id][roi_name]["classes"].append(class_name)
            depths = compute_prediction_depths(cut, np.array(predictions[ovary_id][roi_name]["bboxes"]), resolution=10)
            predictions[ovary_id][roi_name]["depths"] = depths.tolist()
        # break
    return predictions


if __name__ == "__main__":
    with open(DATA_SPLIT_PATH, "r") as f:
        data_split = json.load(f)

    predictions = predict_detr(
        config_path=CONFIG_FILE,
        checkpoint_path=CHECKPOINT_PATH,
        patch_size=PATCH_SIZE,
        ovary_path=OVARY_PATH,
        data_split=data_split,
        class_name_values=CLASS_NAME_VALUES,
        bboxes_size_params=BBOXES_SIZE_PARAMS
    )
    with open(SAVE_PREDICTIONS_PATH, "w") as f:
        json.dump(predictions, f)
