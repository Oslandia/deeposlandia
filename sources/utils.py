""" Utilitary function for Mapillary dataset analysis

# Author: Raphael Delhome
# Organization: Oslandia
# Date: september 2017
"""

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

def compute_monotonic_weights(nb_images, label_counter, mu=0.5, max_weight=10):
    """Compute monotonic weights regarding the popularity of each label given
    by `label_counter`, over a total population of `nb_images`

    Parameters
    ----------
    nb_images: integer
        Number of images over which the weights must be computed
    label_counter: list
        Number of images where each label does appear
    mu: float
        Constant coefficient between 0 and 1
    max_weight: integer
        Maximum weight to apply when counter is too small with respect to
    nb_images (in such a case, the function can give a far too large number)
    """
    return [min(math.log(1 + mu * nb_images / l), max_weight) for l in label_counter]

def compute_centered_weights(nb_images, label_counter, mu=0.5):
    """Compute weights regarding the popularity of each label given by
    `label_counter`, over a total population of `nb_images`; the weights will
    be larger when popularity is either too small or too large (comparison with
    a 50% popularity)

    Parameters
    ----------
    nb_images: integer
        Number of images over which the weights must be computed
    label_counter: list
        Number of images where each label does appear
    mu: float
        Constant coefficient between 0 and 1
    
    """
    return [math.log(1 + mu * (l - nb_images / 2) ** 2 / nb_images)
            for l in label_counter]

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

def mapillary_image_size_plot(data, filename):
    """Plot the distribution of the sizes in a bunch of images, as a hexbin
    plot; the image data are stored into a pandas.DataFrame that contains two
    columns `height` and `width`

    Parameters
    ----------
    data: pd.DataFrame
        image data, with at least two columns `width` and `height`
    filename: object
        string designing the name of the .png file in which is saved the plot
    
    """
    data.plot.hexbin(x="width", y="height", gridsize=25, bins='log')
    plt.plot(data.width, data.height, 'b+', ms=0.75)
    plt.legend(['images'], loc=2)
    plt.plot([0, 8000], [0, 6000], 'r-', linestyle="dashed", linewidth=0.5)
    plt.xlim(0, 7000)
    plt.ylim(0, 5500)
    plt.axvline(x=3264, color="grey", linewidth=0.5, linestyle="dotted")
    plt.axhline(y=2448, color="grey", linewidth=0.5, linestyle="dotted")
    plt.text(6000, 5000, "4:3", color="red")
    plt.text(3500, 600, "width=3264", color="grey")
    plt.text(700, 2600, "height=2448", color="grey")
    plt.title("Number of images with respect to dimensions (log10-scale)",
              fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join("..", "images", filename))

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

def prepare_folders(datapath, dataset, aggregate_value, image_size, model):
    """Data path and repository management ; create all the folders needed to accomplish the
    current instance training/testing

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
    model : str
        Research problem that is tackled, *e.g.* `feature_detection` or `semantic_segmentation`

    Returns
    -------
    dict
        All the meaningful folders and dataset configuration file as a dictionary
    """
    dataset_repo = os.path.join(datapath, dataset)
    input_repo = os.path.join(dataset_repo, "input")
    preprocessed_repo = str(image_size) + "_" + aggregate_value
    preprocessed_path = os.path.join(dataset_repo, "preprocessed", preprocessed_repo)
    training_filename = os.path.join(preprocessed_path, "training.json")
    validation_filename = os.path.join(preprocessed_path, "validation.json")
    testing_filename = os.path.join(preprocessed_path, "testing.json")
    preprocessed_training_path = os.path.join(preprocessed_path, "training")
    preprocessed_validation_path = os.path.join(preprocessed_path, "validation")
    preprocessed_testing_path = os.path.join(preprocessed_path, "testing")
    backup_path = os.path.join(dataset_repo, "output", model)
    os.makedirs(os.path.join(preprocessed_training_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(preprocessed_training_path, "labels"), exist_ok=True)
    os.makedirs(os.path.join(preprocessed_validation_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(preprocessed_validation_path, "labels"), exist_ok=True)
    os.makedirs(os.path.join(preprocessed_testing_path, "images"), exist_ok=True)
    os.makedirs(backup_path, exist_ok=True)
    return {"input": input_repo,
            "prepro_training": preprocessed_training_path,
            "prepro_validation": preprocessed_validation_path,
            "prepro_testing": preprocessed_testing_path,
            "training_config": training_filename,
            "validation_config": validation_filename,
            "testing_config": testing_filename,
            "output": backup_path}
