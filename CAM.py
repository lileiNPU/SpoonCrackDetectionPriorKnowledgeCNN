import os
import cv2
import numpy as np
import torch


def show_CAM(save_img_path, image, feature_maps, class_id, all_ids=10, image_size=(320, 640), normalization=True):
    """
    save_img_path: save heatmap images path
    feature_maps: this is a list [tensor,tensor,tensor], tensor shape is [1, 3, N, N, all_ids]
    normalization: Normalize score and class to 0 to 1
    image_size: w, h
    """
    SHOW_NAME = ["score", "class", "class*score"]
    img_ori = image
    layers0 = feature_maps[0].reshape([-1, all_ids])
    layers1 = feature_maps[1].reshape([-1, all_ids])
    layers2 = feature_maps[2].reshape([-1, all_ids])
    layers = torch.cat([layers0, layers1, layers2], 0)
    if normalization:
        score_max_v = 1.
        score_min_v = 0.
        class_max_v = 1.
        class_min_v = 0.
    else:
        score_max_v = layers[:, 4].max()  # compute max of score from all anchor
        score_min_v = layers[:, 4].min()  # compute min of score from all anchor
        class_max_v = layers[:, 5 + class_id].max()  # compute max of class from all anchor
        class_min_v = layers[:, 5 + class_id].min()  # compute min of class from all anchor
    for j in range(3):  # layers
        layer_one = feature_maps[j]
        # compute max of score from three anchor of the layer
        if normalization:
            anchors_score_max = layer_one[0, :, :, :, 4].max(0)[0].sigmoid()
            # compute max of class from three anchor of the layer
            anchors_class_max = layer_one[0, :, :, :, 5 + class_id].max(0)[0].sigmoid()
        else:
            anchors_score_max = layer_one[0, :, :, :, 4].max(0)[0]
            # compute max of class from three anchor of the layer
            anchors_class_max = layer_one[0, :, :, :, 5 + class_id].max(0)[0]

        scores = ((anchors_score_max - score_min_v) / (
                score_max_v - score_min_v))
        classes = ((anchors_class_max - class_min_v) / (
                class_max_v - class_min_v))

        layer_one_list = []
        layer_one_list.append(scores)
        layer_one_list.append(classes)
        layer_one_list.append(scores * classes)
        for idx, one in enumerate(layer_one_list):
            layer_one = one.cpu().numpy()
            if normalization:
                ret = ((layer_one - layer_one.min()) / (layer_one.max() - layer_one.min())) * 255
            else:
                ret = ((layer_one - 0.) / (1. - 0.)) * 255
            ret = ret.astype(np.uint8)
            gray = ret[:, :, None]
            ret = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

            ret = cv2.resize(ret, image_size)
            img_ori = cv2.resize(img_ori, image_size)

            show = ret * 0.50 + img_ori * 0.50
            show = show.astype(np.uint8)
            import time
            # print(str(time.time()))
            cv2.imwrite(os.path.join(save_img_path, f"{str(time.time())}_{j}_{SHOW_NAME[idx]}.jpg"), show)

# show_CAM(path, ret[1], 21)
