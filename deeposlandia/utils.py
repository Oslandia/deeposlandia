""" Utilitary function for Mapillary dataset analysis
"""

import json
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import re
import sys

# Define the logger for the current project
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(module)s :: %(funcName)s : %(message)s')
ch_stdout = logging.StreamHandler(sys.stdout)
os.makedirs("../log", exist_ok=True)
ch_logfile = logging.FileHandler("../log/cnn_log.log")
ch_stdout.setFormatter(formatter)
ch_logfile.setFormatter(formatter)
logger.addHandler(ch_stdout)
logger.addHandler(ch_logfile)

def read_config(filename):
    """Read the JSON configuration file.

    Parameters
    ----------
    filename : str
        Path of the configuration file

    Returns
    -------
    dict
        Dataset glossary
    """
    with open(filename) as fobj:
        return json.load(fobj)

def mapillary_label_building(filtered_image, label_ids):
    """Build a list of integer labels that are contained into a candidate
    filtered image; according to its pixels

    Parameters
    ----------
    filtered_image : np.array
        Image to label, under the numpy.array format
    label_ids : list
        List of labels ids contained into the reference classification

    Returns
    -------
    dict
        label ids occur or not in the image
    """
    image_data = np.array(filtered_image)
    available_labels = np.unique(image_data)
    return {i: 1 if i in available_labels else 0 for i in label_ids}

def resize_image(img, base_size):
    """ Resize image `img` such that min(width, height)=base_size; keep image
    proportions

    Parameters:
    -----------
    img: Image
        input image to resize
    base_size: integer
        minimal dimension of the returned image
    """
    old_width, old_height = img.size
    if old_width < old_height:
        new_size = (base_size, int(base_size * old_height / old_width))
    else:
        new_size = (int(base_size * old_width / old_height), base_size)
    return img.resize(new_size)

def mono_crop_image(img, crop_pixel):
    """Crop image `img` so as its dimensions become equal (width=height),
    without modifying its smallest dimension
    """
    if img.width > img.height:
        return img.crop((crop_pixel, 0, crop_pixel+img.height, img.height))
    else:
        return img.crop((0, crop_pixel, img.width, crop_pixel+img.width))

def flip_image(img, proba=0.5):
    """ Flip image `img` horizontally with a probability of `proba`

    Parameters:
    -----------
    img: Image
        input image to resize
    proba: float
        probability of flipping input image (if less than 0.5, no flipping)
    """
    if np.random.sample() < proba:
        return img
    else:
        return img.transpose(Image.FLIP_LEFT_RIGHT)

def list_to_str(seq, sep='-'):
    """Transform the input sequence into a ready-to-print string

    Parameters
    ----------
    seq : list, tuple, dict
        Input sequence that must be transformed
    sep : str
        Separator that must appears between each `seq` items

    Returns
    -------
    str
        Printable version of input list
    """
    return sep.join(str(i) for i in seq)

def prepare_input_folder(datapath, dataset):
    """Data path and repository management; create and return the raw dataset path

    Parameters
    ----------
    datapath : str
        Data root directory, contain all used the datasets
    dataset : str
        Dataset name, *e.g.* `mapillary` or `shapes`

    Returns
    -------
    str
        Dataset raw image path

    """
    input_folder = os.path.join(datapath, dataset, "input")
    os.makedirs(input_folder, exist_ok=True)
    return input_folder

def prepare_preprocessed_folder(datapath, dataset, image_size, aggregate_value):
    """Data path and repository management; create all the folders needed to
    accomplish the current instance training/testing

    Parameters
    ----------
    datapath : str
        Data root directory, contain all used the datasets
    dataset : str
        Dataset name, *e.g.* `mapillary` or `shapes`
    aggregate_value : str
        Indicates the label aggregation status, either `full` or `aggregated`
    image_size : int
        Size of the considered images (height and width are equal)

    Returns
    -------
    dict
        All the meaningful folders and dataset configuration file as a dictionary

    """
    prepro_folder = os.path.join(datapath, dataset, "preprocessed",
                                       str(image_size) + "_" + aggregate_value)
    training_filename = os.path.join(prepro_folder, "training.json")
    validation_filename = os.path.join(prepro_folder, "validation.json")
    testing_filename = os.path.join(prepro_folder, "testing.json")
    training_folder = os.path.join(prepro_folder, "training")
    validation_folder = os.path.join(prepro_folder, "validation")
    testing_folder = os.path.join(prepro_folder, "testing")
    os.makedirs(os.path.join(training_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(training_folder, "labels"), exist_ok=True)
    os.makedirs(os.path.join(validation_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(validation_folder, "labels"), exist_ok=True)
    os.makedirs(os.path.join(testing_folder, "images"), exist_ok=True)
    return {"training": training_folder,
            "validation": validation_folder,
            "testing": testing_folder,
            "training_config": training_filename,
            "validation_config": validation_filename,
            "testing_config": testing_filename}

def prepare_output_folder(datapath, dataset, model, instance_name=None):
    """Dataset and repository management; create and return the dataset output path

    Parameters
    ----------
    datapath : str
        Data root directory, contain all used the datasets
    dataset : str
        Dataset name, *e.g.* `mapillary` or `shapes`
    model : str
        Research problem that is tackled, *e.g.* `feature_detection` or `semantic_segmentation`
    instance_name : str
        Instance name, used to create the accurate output folders

    Returns
    -------
    str
        Dataset output path
    """
    if not instance_name is None:
        output_folder = os.path.join(datapath, dataset, "output",
                                     model, "checkpoints", instance_name)
    else:
        output_folder = os.path.join(datapath, dataset, "output",
                                     model, "checkpoints")
    os.makedirs(output_folder, exist_ok=True)
    return output_folder
