from __future__ import print_function
from typing import List
import numpy as np
import cv2


def reformat_input(image):
    if type(image) == np.ndarray:
        if len(image.shape) == 2:  # grayscale
            img_cv_grey = image
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3:  # BGRscale
            img = image
            img_cv_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBAscale
            img = image[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        print('Invalid input type. Suppoting format = string(file path or url), bytes, numpy array')
    return img, img_cv_grey


def reformat_input_list(image_list: List[np.ndarray]):
    img_list = []
    img_cv_grey_list = []
    for image in image_list:
        img, img_cv_grey = reformat_input(image)
        img_list.append(img)
        img_cv_grey_list.append(img_cv_grey)
    return img_list, img_cv_grey_list
