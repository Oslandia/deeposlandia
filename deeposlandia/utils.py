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

def label_building(filtered_image, label_ids, dataset='mapillary'):
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
    if dataset == 'aerial':
        available_labels[available_labels==255] = 1
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

def recover_instance(instance_path):
    """Recover instance specification for model design, *i.e.* dropout rate and network
    architecture

    Parameters
    ----------
    instance_path : str
        Path of the instance specifications on the file system

    Returns
    -------
    tuple
        Instance dropout and network architecture
    """
    with open(instance_path) as fobj:
        instance = json.load(fobj)
    return instance["dropout"], instance["network"]


def GetHTMLColor(color):
    """Convert single-channeled colors or (R, G, B) tuples to #RRGGBB.

    If the color has only one channel (grayscale, typically), it is converted
    to a (R, G, B) tuple first.

    Parameters
    ----------
    color : int or list
        Grayscale value or list of red, green, blue pixel values

    Returns
    -------
    str
        HTML-version of color
    """
    rgb_tuple = (color, color, color) if type(color) == int else color
    return '#%02x%02x%02x' % tuple(rgb_tuple)


def build_image_from_config(img, config):
    """Rebuild a labelled image version from dataset configuration

    Parameters
    ----------
    img : PIL.Image
        Labelled image ; squared-size, shape (imsize, imsize)
    config : dict
        Label definition and description

    Returns
    -------
    np.array
        RGB-version of labelled images, with shape (imsize, imsize, 3)
    """
    data = np.array(img)
    result = np.zeros(shape=(data.shape[0], data.shape[1], 3),
                      dtype=data.dtype)
    for label in range(len(config)):
        mask = data == label
        result[mask] = config[label]["color"]
    return Image.fromarray(result)


def create_symlink(link_name, directory):
    """Create a symbolic link

    Based on https://github.com/elemoine/dotfiles/blob/master/install-dotfiles.py

    Parameters
    ----------
    link_name : string
        name of the symbolic link
    directory : string
        Targetted destination for the symbolic link
    """
    if os.path.islink(link_name):
        os.unlink(link_name)
    elif os.path.isfile(link_name):
        logger.error("{} is a file!".format(link_name))
    elif os.path.isdir(link_name):
        logger.error("{} is a directory!".format(link_name))
    os.symlink(directory, link_name)
    logger.info("{} now points to {}.".format(link_name, directory))


def tile_image_correspondance(raw_img_size):
    """Provides a table of association between possible image sizes and tile
    size, in the context of Aerial image dataset.

    In such a dataset, large `raw_img_size`*`raw_img_size` images are tiled so
    as to exploit smaller images during neural network training. The point is
    to find the image size that fits the best the input tile size.

    Tile sizes are chosen in order to use every original image pixel, then they
    are 5000-divisors. Whilst image sizes are chosen to ensure a deep neural
    network training, *i.e.* they are divisible per 16 (to allow until four
    encoding layers).

    As this relation is not bijective (each tile size gives one single image
    size, but some image size gives several possible tile sizes), we
    hypothesize that the best tile is the smallest one. This choice is made
    relatively to image preprocessing, as cropping/resizing step will modify
    the raw tiles: we prefer to minimize these image distortions.

    Parameters
    ----------
    raw_img_size : int
        Size of tiled image

    Returns
    -------
    pd.DataFrame
        Association table

    """
    associations = [[(j, i) for j in range(0, i, 16)][-1]
                    for i in range(1, raw_img_size)
                    if raw_img_size % i == 0]
    df = pd.DataFrame(associations, columns=["m16", "d5000"])
    return df.groupby("m16").min().reset_index()


def get_image_size_from_tile(tile_size, raw_img_size=5000):
    """Give the biggest 16-multiplier that is smaller than `tile_size`

    This method aims at converting a given image tile size into a new
    exploitable size (*i.e.* a size compatible with neural network layer
    operations).

    Parameters
    ----------
    tile_size : int
        Size of the input tiles
    raw_img_size : int
        Size of tiled image

    Returns
    -------
    int
        Output size compatible with neural network layer operations

    """
    aerial_table = tile_image_correspondance(raw_img_size)
    image_size = aerial_table.loc[aerial_table.d5000 == tile_size, "m16"]
    if len(image_size.index) == 0:
        raise ValueError(("There is no value in the association table that "
                          "corresponds to this tile size ({}). Please choose "
                          "one of these values: {}"
                          "").format(tile_size, aerial_table.d5000.values))
    return image_size.item()


def get_tile_size_from_image(image_size, raw_img_size=5000):
    """Retrieve the original dataset tile size knowing that `image_size` is the
    biggest 16-multiplier that is smaller than it; we hypothesize that the tile
    size is a divisor of 5000 (the original aerial dataset image size)

    This method aims at converting back an exploitable image size (*i.e.* a
    size compatible with neural network layer operations) to the tile size used
    when the dataset was created.

    Parameters
    ----------
    image_size : int
        Size of the image (compatible with neural network layer operations)
    raw_img_size : int
        Size of tiled image

    Returns
    -------
    int
        Tile size used for preprocessed dataset building

    """
    aerial_table = tile_image_correspondance(raw_img_size)
    tile_size = aerial_table.loc[aerial_table.m16 == image_size, "d5000"]
    if len(tile_size.index) == 0:
        raise ValueError(("There is no value in the association table that "
                          "corresponds to this image size ({}). Please choose "
                          "one of these values: {}"
                          "").format(image_size, aerial_table.m16.values))
    return tile_size.item()
