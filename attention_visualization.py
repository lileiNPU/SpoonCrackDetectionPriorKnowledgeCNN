import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from sklearn import preprocessing
from tqdm import tqdm
import sys
sys.path.append('.')
from od.models.modules.experimental import attempt_load
from od import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized
import numpy as np
import cv2


def visualize_attention_map(attention_map):
    """
    The attention map is a matrix ranging from 0 to 1, where the greater the value,
    the greater attention is suggests.
    :param attention_map: np.numpy matrix hanging from 0 to 1
    :return np.array matrix with rang [0, 255]
    """
    attention_map_color = np.zeros(
        shape=[attention_map.shape[0], attention_map.shape[1], 3],
        dtype=np.uint8
    )

    red_color_map = np.zeros(
        shape=[attention_map.shape[0], attention_map.shape[1]],
        dtype=np.uint8
    ) + 255

    red_color_map = red_color_map * attention_map
    red_color_map = np.array(red_color_map, dtype=np.uint8)

    attention_map_color[:, :, 0] = red_color_map
    # attention_map_color[:, :, 1] = red_color_map
    # attention_map_color[:, :, 2] = red_color_map
    return attention_map_color


model = attempt_load('E:/Research-code/Spoon-yolo/flexible-yolov5-main/result/weights/best.pt', map_location='cuda')  # load FP32 model
print(model)
for name, param in model.named_parameters():
    print("-----model.named_parameters()--{}:{}".format(name, ""))
    if name == 'backbone.attentionmodule_1.bottleneck1_1.conv1.weight':
        # print("-----model.param()--{}:{}".format(param, ""))
        print(param.shape)
        param1 = param.squeeze(dim=2).squeeze(dim=2).clone()
        param1 = param1.cpu().numpy()
        # cv2.imshow('GrayImage', visualize_attention_map(param1))
        # param1 = visualize_attention_map(param1)


        # heatmap[:,:,2] = 0
        heatmap = param1
        print(heatmap)

        min_max_scaler_get_fitness = preprocessing.MinMaxScaler()
        heatmap = min_max_scaler_get_fitness.fit_transform(heatmap)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img_path = 'E:/Research-code/Spoon-yolo/flexible-yolov5-main/datasets/images/11129.jpg'

        img = cv2.imread(img_path)
        mask = heatmap
        cv2.imshow('imgadd', cv2.resize(mask, (800, 800)))
        cv2.waitKey()
        # 叠加显示img, mask

        # imgadd = cv2.add(cv2.resize(img, (800, 800)), cv2.resize(mask, (800, 800)))
        # cv2.imshow('imgadd', imgadd)
        cv2.waitKey()

