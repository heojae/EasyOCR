# -*- coding: utf-8 -*-
from typing import List

from .detection import get_detector, get_textbox
from .recognition import get_recognizer, get_text
from .utils import group_text_box, get_image_list, calculate_md5, get_paragraph, \
    download_and_unzip, printProgressBar, diff, reformat_input
import numpy as np
import cv2
import torch
import os
import sys
import math
from PIL import Image
from logging import getLogger

#############################################
from .custom_utils import reformat_input_list
from .custom_detection import batch_test_net, batch_get_textbox
from .recognition import AlignCollate, recognizer_predict, ListDataset

#############################################

def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def get_all_cropped_image_and_max_width(batch_horizontal_list, batch_free_list, img_cv_grey_list, model_height=64):
    crop_image_list = []

    max_width = 1
    imgH = 64
    for i in range(0, len(batch_horizontal_list)):
        img_cv_grey = img_cv_grey_list[i]
        horizontal_list = batch_horizontal_list[i]
        free_list = batch_free_list[i]

        temp_maximum_y, temp_maximum_x = img_cv_grey.shape
        max_ratio_hori, max_ratio_free = 1, 1
        temp_image_list = []

        if (horizontal_list == None) and (free_list == None):
            y_max, x_max = img_cv_grey.shape
            ratio = x_max / y_max
            temp_max_width = int(imgH * ratio)
            crop_img = cv2.resize(img_cv_grey, (max_width, imgH), interpolation=Image.ANTIALIAS)
            temp_image_list = [([[0, 0], [x_max, 0], [x_max, y_max], [0, y_max]], crop_img, i)]
            if temp_max_width > max_width:
                max_width = temp_max_width
            crop_image_list.extend(temp_image_list)
            continue

        for box in free_list:
            rect = np.array(box, dtype="float32")
            transformed_img = four_point_transform(img_cv_grey, rect)
            ratio = transformed_img.shape[1] / transformed_img.shape[0]
            crop_img = cv2.resize(transformed_img, (int(model_height * ratio), model_height),
                                  interpolation=Image.ANTIALIAS)
            temp_image_list.append((box, crop_img, i))  # box = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            max_ratio_free = max(ratio, max_ratio_free)
        max_ratio_free = math.ceil(max_ratio_free)

        for box in horizontal_list:
            x_min = max(0, box[0])
            x_max = min(box[1], temp_maximum_x)
            y_min = max(0, box[2])
            y_max = min(box[3], temp_maximum_y)
            crop_img = img_cv_grey[y_min: y_max, x_min:x_max]
            width = x_max - x_min
            height = y_max - y_min
            ratio = width / height
            crop_img = cv2.resize(crop_img, (int(model_height * ratio), model_height), interpolation=Image.ANTIALIAS)
            temp_image_list.append(([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], crop_img, i))
            max_ratio_hori = max(ratio, max_ratio_hori)

        max_ratio_hori = math.ceil(max_ratio_hori)
        max_ratio = max(max_ratio_hori, max_ratio_free)
        temp_max_width = math.ceil(max_ratio) * model_height
        if temp_max_width > max_width:
            max_width = temp_max_width
        temp_image_list = sorted(temp_image_list, key=lambda item: item[0][0][1])  # sort by vertical position
        crop_image_list.extend(temp_image_list)
    return crop_image_list, max_width







def batch_get_text(character, imgH, imgW, recognizer, converter, image_list, \
             ignore_char='', decoder='greedy', beamWidth=5, batch_size=1, contrast_ths=0.1, \
             adjust_contrast=0.5, filter_ths=0.003, workers=1, device='cpu'):
    batch_max_length = int(imgW / 10)

    char_group_idx = {}
    ignore_idx = []
    for char in ignore_char:
        try:
            ignore_idx.append(character.index(char) + 1)
        except:
            pass

    coord = [item[0] for item in image_list]
    img_list = [item[1] for item in image_list]
    cut_point_list = [item[2] for item in image_list]
    AlignCollate_normal = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True)
    test_data = ListDataset(img_list)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=int(workers), collate_fn=AlignCollate_normal, pin_memory=True)

    # predict first round
    result1 = recognizer_predict(recognizer, converter, test_loader, batch_max_length, \
                                 ignore_idx, char_group_idx, decoder, beamWidth, device=device)

    # predict second round
    low_confident_idx = [i for i, item in enumerate(result1) if (item[1] < contrast_ths)]
    if len(low_confident_idx) > 0:
        img_list2 = [img_list[i] for i in low_confident_idx]
        AlignCollate_contrast = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True,
                                             adjust_contrast=adjust_contrast)
        test_data = ListDataset(img_list2)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=False,
            num_workers=int(workers), collate_fn=AlignCollate_contrast, pin_memory=True)
        result2 = recognizer_predict(recognizer, converter, test_loader, batch_max_length, \
                                     ignore_idx, char_group_idx, decoder, beamWidth, device=device)

    result = []
    for i, zipped in enumerate(zip(coord, result1)):
        box, pred1 = zipped
        if i in low_confident_idx:
            pred2 = result2[low_confident_idx.index(i)]
            if pred1[1] > pred2[1]:
                result.append((box, pred1[0], pred1[1], cut_point_list[i]))
            else:
                result.append((box, pred2[0], pred2[1], cut_point_list[i]))
        else:
            result.append((box, pred1[0], pred1[1], cut_point_list[i]))

    # confidence_score = pred_max_prob.cumprod(dim=0)[-1]
    # if confidence_score.item() > filter_ths:
    #    print(pred, confidence_score.item())
    # else:
    #    print('not sure', pred, confidence_score.item())

    return result, cut_point_list



















