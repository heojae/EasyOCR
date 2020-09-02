# -*- coding: utf-8 -*-
from typing import List

from .detection import get_detector, get_textbox
from .recognition import get_recognizer, get_text
from .utils import group_text_box, get_image_list, calculate_md5, get_paragraph, \
    download_and_unzip, printProgressBar, diff, reformat_input
from bidi.algorithm import get_display
import numpy as np
import cv2
import torch
import os
import sys
from PIL import Image
from logging import getLogger

if sys.version_info[0] == 2:
    from io import open
    from six.moves.urllib.request import urlretrieve
    from pathlib2 import Path
else:
    from urllib.request import urlretrieve
    from pathlib import Path
#############################################
from .custom_utils import reformat_input_list
from .custom_detection import batch_test_net, batch_get_textbox

#############################################


os.environ["LRU_CACHE_CAPACITY"] = "1"
LOGGER = getLogger(__name__)

BASE_PATH = os.path.dirname(__file__)
MODULE_PATH = os.environ.get("EASYOCR_MODULE_PATH") or \
              os.environ.get("MODULE_PATH") or \
              os.path.expanduser("~/.EasyOCR/")

# detector parameters
DETECTOR_FILENAME = 'craft_mlt_25k.pth'

# recognizer parameters
latin_lang_list = ['af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'en', 'es', 'et', 'fr', 'ga', \
                   'hr', 'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', \
                   'nl', 'no', 'oc', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', \
                   'sv', 'sw', 'tl', 'tr', 'uz', 'vi']
arabic_lang_list = ['ar', 'fa', 'ug', 'ur']
bengali_lang_list = ['bn', 'as']
cyrillic_lang_list = ['ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', \
                      'ava', 'dar', 'inh', 'che', 'lbe', 'lez', 'tab']
devanagari_lang_list = ['hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom']

all_lang_list = latin_lang_list + arabic_lang_list + cyrillic_lang_list + devanagari_lang_list + bengali_lang_list + [
    'th', 'ch_sim', 'ch_tra', 'ja', 'ko', 'ta']
imgH = 64
input_channel = 1
output_channel = 512
hidden_size = 512

number = '0123456789'
symbol = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
en_char = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

# first element is url path, second is file size
model_url = {
    'detector': ('https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip',
                 '2f8227d2def4037cdb3b34389dcf9ec1'),
    'latin.pth': (
        'https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/latin.zip',
        'fb91b9abf65aeeac95a172291b4a6176'),
    'chinese.pth': (
        'https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/chinese.zip',
        'dfba8e364cd98ed4fed7ad54d71e3965'),
    'chinese_sim.pth': ('https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/chinese_sim.zip',
                        '0e19a9d5902572e5237b04ee29bdb636'),
    'japanese.pth': ('https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/japanese.zip',
                     '6d891a4aad9cb7f492809515e4e9fd2e'),
    'korean.pth': (
        'https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/korean.zip',
        '45b3300e0f04ce4d03dda9913b20c336'),
    'thai.pth': (
        'https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/thai.zip',
        '40a06b563a2b3d7897e2d19df20dc709'),
    'devanagari.pth': ('https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/devanagari.zip',
                       'db6b1f074fae3070f561675db908ac08'),
    'cyrillic.pth': ('https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/cyrillic.zip',
                     '5a046f7be2a4f7da6ed50740f487efa8'),
    'arabic.pth': (
        'https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/arabic.zip',
        '993074555550e4e06a6077d55ff0449a'),
    'tamil.pth': (
        'https://github.com/JaidedAI/EasyOCR/releases/download/v1.1.7/tamil.zip', '4b93972fdacdcdabe6d57097025d4dc2'),
    'bengali.pth': (
        'https://github.com/JaidedAI/EasyOCR/releases/download/v1.1.8/bengali.zip', 'cea9e897e2c0576b62cbb1554997ce1c'),
}


class CustomReader(object):

    def __init__(self, lang_list, gpu=True, model_storage_directory=None,
                 download_enabled=True, detector=True, recognizer=True):
        """Create an EasyOCR Reader.

        Parameters:
            lang_list (list): Language codes (ISO 639) for languages to be recognized during analysis.

            gpu (bool): Enable GPU support (default)

            model_storage_directory (string): Path to directory for model data. If not specified,
            models will be read from a directory as defined by the environment variable
            EASYOCR_MODULE_PATH (preferred), MODULE_PATH (if defined), or ~/.EasyOCR/.

            download_enabled (bool): Enabled downloading of model data via HTTP (default).
        """
        self.download_enabled = download_enabled

        self.model_storage_directory = MODULE_PATH + '/model'
        if model_storage_directory:
            self.model_storage_directory = model_storage_directory
        Path(self.model_storage_directory).mkdir(parents=True, exist_ok=True)

        if gpu is False:
            self.device = 'cpu'
            LOGGER.warning('Using CPU. Note: This module is much faster with a GPU.')
        elif not torch.cuda.is_available():
            self.device = 'cpu'
            LOGGER.warning('CUDA not available - defaulting to CPU. Note: This module is much faster with a GPU.')
        elif gpu is True:
            self.device = 'cuda'
        else:
            self.device = gpu

        # check available languages
        unknown_lang = set(lang_list) - set(all_lang_list)
        if unknown_lang != set():
            raise ValueError(unknown_lang, 'is not supported')

        # choose model
        if 'th' in lang_list:
            self.model_lang = 'thai'
            if set(lang_list) - set(['th', 'en']) != set():
                raise ValueError('Thai is only compatible with English, try lang_list=["th","en"]')
        elif 'ch_tra' in lang_list:
            self.model_lang = 'chinese_tra'
            if set(lang_list) - set(['ch_tra', 'en']) != set():
                raise ValueError('Chinese is only compatible with English, try lang_list=["ch_tra","en"]')
        elif 'ch_sim' in lang_list:
            self.model_lang = 'chinese_sim'
            if set(lang_list) - set(['ch_sim', 'en']) != set():
                raise ValueError('Chinese is only compatible with English, try lang_list=["ch_sim","en"]')
        elif 'ja' in lang_list:
            self.model_lang = 'japanese'
            if set(lang_list) - set(['ja', 'en']) != set():
                raise ValueError('Japanese is only compatible with English, try lang_list=["ja","en"]')
        elif 'ko' in lang_list:
            self.model_lang = 'korean'
            if set(lang_list) - set(['ko', 'en']) != set():
                raise ValueError('Korean is only compatible with English, try lang_list=["ko","en"]')
        elif 'ta' in lang_list:
            self.model_lang = 'tamil'
            if set(lang_list) - set(['ta', 'en']) != set():
                raise ValueError('Tamil is only compatible with English, try lang_list=["ta","en"]')
        elif set(lang_list) & set(bengali_lang_list):
            self.model_lang = 'bengali'
            if set(lang_list) - set(bengali_lang_list + ['en']) != set():
                raise ValueError('Bengali is only compatible with English, try lang_list=["bn","as","en"]')
        elif set(lang_list) & set(arabic_lang_list):
            self.model_lang = 'arabic'
            if set(lang_list) - set(arabic_lang_list + ['en']) != set():
                raise ValueError('Arabic is only compatible with English, try lang_list=["ar","fa","ur","ug","en"]')
        elif set(lang_list) & set(devanagari_lang_list):
            self.model_lang = 'devanagari'
            if set(lang_list) - set(devanagari_lang_list + ['en']) != set():
                raise ValueError('Devanagari is only compatible with English, try lang_list=["hi","mr","ne","en"]')
        elif set(lang_list) & set(cyrillic_lang_list):
            self.model_lang = 'cyrillic'
            if set(lang_list) - set(cyrillic_lang_list + ['en']) != set():
                raise ValueError(
                    'Cyrillic is only compatible with English, try lang_list=["ru","rs_cyrillic","be","bg","uk","mn","en"]')
        else:
            self.model_lang = 'latin'

        separator_list = {}
        if self.model_lang == 'latin':
            all_char = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' + \
                       'ÀÁÂÃÄÅÆÇÈÉÊËÍÎÑÒÓÔÕÖØÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿąęĮįıŁłŒœŠšųŽž'
            self.character = number + symbol + all_char
            model_file = 'latin.pth'
        elif self.model_lang == 'arabic':
            ar_number = '٠١٢٣٤٥٦٧٨٩'
            ar_symbol = '«»؟،؛'
            ar_char = 'ءآأؤإئااًبةتثجحخدذرزسشصضطظعغفقكلمنهوىيًٌٍَُِّْٰٓٔٱٹپچڈڑژکڭگںھۀہۂۃۆۇۈۋیېےۓە'
            self.character = number + symbol + en_char + ar_number + ar_symbol + ar_char
            model_file = 'arabic.pth'
        elif self.model_lang == 'cyrillic':
            cyrillic_char = 'ЁЂЄІЇЈЉЊЋЎЏАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёђєіїјљњћўџҐґҮүө'
            self.character = number + symbol + en_char + cyrillic_char
            model_file = 'cyrillic.pth'
        elif self.model_lang == 'devanagari':
            devanagari_char = '.ँंःअअंअःआइईउऊऋएऐऑओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळवशषसह़ािीुूृॅेैॉोौ्ॐ॒क़ख़ग़ज़ड़ढ़फ़ॠ।०१२३४५६७८९॰'
            self.character = number + symbol + en_char + devanagari_char
            model_file = 'devanagari.pth'
        elif self.model_lang == 'bengali':
            bn_char = '।ঁংঃঅআইঈউঊঋঌএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ািীুূৃেৈোৌ্ৎড়ঢ়য়০১২৩৪৫৬৭৮৯'
            self.character = number + symbol + en_char + bn_char
            model_file = 'bengali.pth'
        elif self.model_lang == 'chinese_tra':
            char_file = os.path.join(BASE_PATH, 'character', "ch_tra_char.txt")
            with open(char_file, "r", encoding="utf-8-sig") as input_file:
                ch_tra_list = input_file.read().splitlines()
                ch_tra_char = ''.join(ch_tra_list)
            self.character = number + symbol + en_char + ch_tra_char
            model_file = 'chinese.pth'
        elif self.model_lang == 'chinese_sim':
            char_file = os.path.join(BASE_PATH, 'character', "ch_sim_char.txt")
            with open(char_file, "r", encoding="utf-8-sig") as input_file:
                ch_sim_list = input_file.read().splitlines()
                ch_sim_char = ''.join(ch_sim_list)
            self.character = number + symbol + en_char + ch_sim_char
            model_file = 'chinese_sim.pth'
        elif self.model_lang == 'japanese':
            char_file = os.path.join(BASE_PATH, 'character', "ja_char.txt")
            with open(char_file, "r", encoding="utf-8-sig") as input_file:
                ja_list = input_file.read().splitlines()
                ja_char = ''.join(ja_list)
            self.character = number + symbol + en_char + ja_char
            model_file = 'japanese.pth'
        elif self.model_lang == 'korean':
            char_file = os.path.join(BASE_PATH, 'character', "ko_char.txt")
            with open(char_file, "r", encoding="utf-8-sig") as input_file:
                ko_list = input_file.read().splitlines()
                ko_char = ''.join(ko_list)
            self.character = number + symbol + en_char + ko_char
            model_file = 'korean.pth'
        elif self.model_lang == 'tamil':
            char_file = os.path.join(BASE_PATH, 'character', "ta_char.txt")
            with open(char_file, "r", encoding="utf-8-sig") as input_file:
                ta_list = input_file.read().splitlines()
                ta_char = ''.join(ta_list)
            self.character = number + symbol + en_char + ta_char
            model_file = 'tamil.pth'
        elif self.model_lang == 'thai':
            separator_list = {
                'th': ['\xa2', '\xa3'],
                'en': ['\xa4', '\xa5']
            }
            separator_char = []
            for lang, sep in separator_list.items():
                separator_char += sep

            special_c0 = 'ุู'
            special_c1 = 'ิีืึ' + 'ั'
            special_c2 = '่้๊๋'
            special_c3 = '็์'
            special_c = special_c0 + special_c1 + special_c2 + special_c3 + 'ำ'
            th_char = 'กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮฤ' + 'เแโใไะา' + special_c + 'ํฺ' + 'ฯๆ'
            th_number = '0123456789๑๒๓๔๕๖๗๘๙'
            self.character = ''.join(separator_char) + symbol + en_char + th_char + th_number
            model_file = 'thai.pth'
        else:
            LOGGER.error('invalid language')

        dict_list = {}
        for lang in lang_list:
            dict_list[lang] = os.path.join(BASE_PATH, 'dict', lang + ".txt")

        self.lang_char = []
        for lang in lang_list:
            char_file = os.path.join(BASE_PATH, 'character', lang + "_char.txt")
            with open(char_file, "r", encoding="utf-8-sig") as input_file:
                char_list = input_file.read().splitlines()
            self.lang_char += char_list
        self.lang_char = set(self.lang_char).union(set(number + symbol))
        self.lang_char = ''.join(self.lang_char)

        model_path = os.path.join(self.model_storage_directory, model_file)
        corrupt_msg = 'MD5 hash mismatch, possible file corruption'
        detector_path = os.path.join(self.model_storage_directory, DETECTOR_FILENAME)
        if os.path.isfile(detector_path) == False:
            if not self.download_enabled:
                raise FileNotFoundError("Missing %s and downloads disabled" % detector_path)
            LOGGER.warning('Downloading detection model, please wait. '
                           'This may take several minutes depending upon your network connection.')
            download_and_unzip(model_url['detector'][0], DETECTOR_FILENAME, self.model_storage_directory)
            assert calculate_md5(detector_path) == model_url['detector'][1], corrupt_msg
            LOGGER.info('Download complete')
        elif calculate_md5(detector_path) != model_url['detector'][1]:
            if not self.download_enabled:
                raise FileNotFoundError("MD5 mismatch for %s and downloads disabled" % detector_path)
            LOGGER.warning(corrupt_msg)
            os.remove(detector_path)
            LOGGER.warning('Re-downloading the detection model, please wait. '
                           'This may take several minutes depending upon your network connection.')
            download_and_unzip(model_url['detector'][0], DETECTOR_FILENAME, self.model_storage_directory)
            assert calculate_md5(detector_path) == model_url['detector'][1], corrupt_msg
        # check model file
        if os.path.isfile(model_path) == False:
            if not self.download_enabled:
                raise FileNotFoundError("Missing %s and downloads disabled" % model_path)
            LOGGER.warning('Downloading recognition model, please wait. '
                           'This may take several minutes depending upon your network connection.')
            download_and_unzip(model_url[model_file][0], model_file, self.model_storage_directory)
            assert calculate_md5(model_path) == model_url[model_file][1], corrupt_msg
            LOGGER.info('Download complete.')
        elif calculate_md5(model_path) != model_url[model_file][1]:
            if not self.download_enabled:
                raise FileNotFoundError("MD5 mismatch for %s and downloads disabled" % model_path)
            LOGGER.warning(corrupt_msg)
            os.remove(model_path)
            LOGGER.warning('Re-downloading the recognition model, please wait. '
                           'This may take several minutes depending upon your network connection.')
            download_and_unzip(model_url[model_file][0], model_file, self.model_storage_directory)
            assert calculate_md5(model_path) == model_url[model_file][1], corrupt_msg
            LOGGER.info('Download complete')
        if detector:
            self.detector = get_detector(detector_path, self.device)
        if recognizer:
            self.recognizer, self.converter = get_recognizer(input_channel, output_channel, \
                                                             hidden_size, self.character, separator_list, \
                                                             dict_list, model_path, device=self.device)

    def detect(self, img_list: List[np.ndarray], min_size=20, text_threshold=0.7, low_text=0.4, \
               link_threshold=0.4, canvas_size=2560, mag_ratio=1., \
               slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5, \
               width_ths=0.5, add_margin=0.1, reformat=True):

        batch_text_box = batch_get_textbox(self.detector, img_list, canvas_size, mag_ratio, \
                                           text_threshold, link_threshold, low_text, \
                                           False, self.device)

        batch_horizontal_list = []
        batch_free_list = []
        for text_box in batch_text_box:
            horizontal_list, free_list = group_text_box(text_box, slope_ths, \
                                                        ycenter_ths, height_ths, \
                                                        width_ths, add_margin)
            batch_horizontal_list.append(horizontal_list)
            batch_free_list.append(free_list)

        # if min_size:
        #     horizontal_list = [i for i in horizontal_list if max(i[1] - i[0], i[3] - i[2]) > min_size]
        #     free_list = [i for i in free_list if max(diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size]

        return batch_horizontal_list, batch_free_list

    def recognize(self, img_cv_grey_list, batch_horizontal_list=None, batch_free_list=None, \
                  decoder='greedy', beamWidth=5, batch_size=1, \
                  workers=0, allowlist=None, blocklist=None, detail=1, \
                  paragraph=False, \
                  contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003, \
                  reformat=True):
        result_list = []
        for i in range(0, len(batch_horizontal_list)):
            img_cv_grey = img_cv_grey_list[i]
            horizontal_list = batch_horizontal_list[i]
            free_list = batch_free_list[i]

            if (horizontal_list == None) and (free_list == None):
                y_max, x_max = img_cv_grey.shape
                ratio = x_max / y_max
                max_width = int(imgH * ratio)
                crop_img = cv2.resize(img_cv_grey, (max_width, imgH), interpolation=Image.ANTIALIAS)
                image_list = [([[0, 0], [x_max, 0], [x_max, y_max], [0, y_max]], crop_img)]
            else:
                image_list, max_width = get_image_list(horizontal_list, free_list, img_cv_grey, model_height=imgH)

            if allowlist:
                ignore_char = ''.join(set(self.character) - set(allowlist))
            elif blocklist:
                ignore_char = ''.join(set(blocklist))
            else:
                ignore_char = ''.join(set(self.character) - set(self.lang_char))

            if self.model_lang in ['chinese_tra', 'chinese_sim', 'japanese', 'korean']: decoder = 'greedy'
            result = get_text(self.character, imgH, int(max_width), self.recognizer, self.converter, image_list, \
                              ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths, \
                              workers, self.device)

            if self.model_lang == 'arabic':
                direction_mode = 'rtl'
                result = [list(item) for item in result]
                for item in result:
                    item[1] = get_display(item[1])
            else:
                direction_mode = 'ltr'

            if paragraph:
                result = get_paragraph(result, mode=direction_mode)

            if detail == 0:
                result = [item[1] for item in result]
                result_list.append(result)
            else:
                result_list.append(result)
        return result_list

    def convert_image_list2resized_image_list(self, image_list, size=512):
        resized_image_list = []
        for image in image_list:
            resized_image = cv2.resize(image, dsize=(size, size))
            resized_image_list.append(resized_image)
        return resized_image_list

    def back_to_original_point(self, image_list, resized_image_list, batch_horizontal_list, batch_free_list):
        for i in range(0, len(image_list)):
            image = image_list[i]
            new_height, new_width, new_channel = image.shape
            old_height, old_width, old_channel = 512, 512, 3
            height_ratio = new_height / old_height
            width_ratio = new_width / old_width

            horizontal_list = batch_horizontal_list[i]
            for j in range(0, len(horizontal_list)):
                [x_min, x_max, y_min, y_max] = horizontal_list[j]
                x_min, x_max = int(x_min * width_ratio), int(x_max * width_ratio)
                y_min, y_max = int(y_min * height_ratio), int(y_max * height_ratio)
                horizontal_list[j] = [x_min, x_max, y_min, y_max]

            free_list = batch_free_list[i]
            for j in range(0, len(free_list)):
                [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = free_list[j]
                x1, x2, x3, x4 = int(x1 * width_ratio), int(x2 * width_ratio), int(x3 * width_ratio), int(x4 * width_ratio)
                y1, y2, y3, y4 = int(y1 * height_ratio), int(y2 * height_ratio), int(y3 * height_ratio), int(y4 * height_ratio)
                free_list[j] = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    def readtext(self, image_list, decoder='greedy', beamWidth=5, batch_size=1, \
                 workers=0, allowlist=None, blocklist=None, detail=1, \
                 paragraph=False, min_size=20, \
                 contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003, \
                 text_threshold=0.7, low_text=0.4, link_threshold=0.4, \
                 canvas_size=2560, mag_ratio=1., \
                 slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5, \
                 width_ths=0.5, add_margin=0.1):
        '''
        Parameters:
        image: file path or numpy-array or a byte stream object
        '''
        img_list, img_cv_grey_list = reformat_input_list(image_list=image_list)

        resized_img_list = self.convert_image_list2resized_image_list(image_list, size=512)

        batch_horizontal_list, batch_free_list = self.detect(resized_img_list, min_size, text_threshold, \
                                                             low_text, link_threshold, \
                                                             canvas_size, mag_ratio, \
                                                             slope_ths, ycenter_ths, \
                                                             height_ths, width_ths, \
                                                             add_margin, False)
        print("batch_horizontal_list", batch_horizontal_list)
        print("len(batch_horizontal_list)", len(batch_horizontal_list))
        print("batch_free_list", batch_free_list)
        print("len(batch_free_list)", len(batch_free_list))

        self.back_to_original_point(image_list, resized_img_list, batch_horizontal_list, batch_free_list)

        result = self.recognize(img_cv_grey_list, batch_horizontal_list, batch_free_list, \
                                decoder, beamWidth, batch_size, \
                                workers, allowlist, blocklist, detail, \
                                paragraph, contrast_ths, adjust_contrast, \
                                filter_ths, False)

        return result
