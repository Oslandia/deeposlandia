# Author: Raphael Delhome
# Organization: Oslandia
# Date: september 2017

# Utilitary function for Mapillary dataset analysis

import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import re
import sys

DATASET = ["training", "validation", "testing"]
IMAGE_TYPES = ["images", "instances", "labels"]

def make_dir(path):
    """ Create a directory if there isn't one already.

    Parameters
    ----------
    path: object
        string corresponding to the relative path from the current working
    space to the directory that has to be created
    
    """
    try:
        os.mkdir(path)
    except OSError:
        pass

# Define the logger for the current project
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
ch_stdout = logging.StreamHandler(sys.stdout)
make_dir("../log")
ch_logfile = logging.FileHandler("../log/cnn_log.log")
ch_stdout.setFormatter(formatter)
ch_logfile.setFormatter(formatter)
logger.addHandler(ch_stdout)
logger.addHandler(ch_logfile)

def extract_features(data, pattern):
    """Extract features from data that respect the given string pattern

    Parameters
    ----------
    data: pd.DataFrame
        dataframe to filter
    pattern: object
        string designing the column filter
    """
    return data[[col for col in data.columns if re.search(pattern, col) is not None]]

def unnest(l):
    """Unnest a list of lists, by splitting sublists and returning a simple list of scalar elements

    Parameters
    ----------
    l: list
        list of lists
    
    """
    return [index for sublist in l for index in sublist]


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

def mapillary_label_reading(labels):
    """Gives the readable versions of Mapillary labels

    Parameters
    ----------
    labels: dict
        set of Mapillary labels, as defined in the Mapillary classification
    
    """
    return [l['readable'] for l in labels]

def mapillary_label_building(filtered_image, nb_labels):
    """Build a list of integer labels that are contained into a candidate
    filtered image; according to its pixels

    Parameters
    ----------
    filtered_image: np.array
        Image to label, under the numpy.array format
    nb_labels: integer
        number of labels contained into the reference classification
    
    """
    filtered_data = np.array(filtered_image)
    avlble_labels = (pd.Series(filtered_data.reshape([-1]))
                     .value_counts()
                     .index)
    return [1 if i in avlble_labels else 0 for i in range(nb_labels)]

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

def size_inventory():
    """Make an inventory of the size of every image in the dataset (training,
    validation and testing sets taken together)

    """
    sizes = []
    widths = []
    heights = []
    types = []
    filenames = []
    datasets = []
    for dataset in DATASET:
        logger.info("Size inventory in {} dataset".format(dataset))
        for img_filename in os.listdir(os.path.join("..", "data", dataset, "images")):
            for img_type in IMAGE_TYPES:
                if dataset == "testing" and not img_type == "images":
                    continue
                complete_filename = os.path.join("..", "data", dataset,
                                                 img_type, img_filename)
                if not img_type == "images":
                    complete_filename = complete_filename.replace("images", img_type)
                    complete_filename = complete_filename.replace(".jpg", ".png")
                image = Image.open(complete_filename)
                datasets.append(dataset)
                types.append(img_type)
                filenames.append(complete_filename.split("/")[-1].split(".")[0])
                sizes.append(image.size)
                widths.append(image.size[0])
                heights.append(image.size[1])
    return pd.DataFrame({"dataset": datasets,
                      "filename": filenames,
                      "img_type": types,
                      "size": sizes,
                      "width": widths,
                      "height": heights})

def mapillary_data_preparation(datapath, dataset, size, nb_labels):
    """Prepare the Mapillary dataset for processing convolutional neural network
    computing: the images are resized and renamed as `index.jpg`, where `index`
    is an integer comprised between 0 and 17999 (size of the validation
    dataset), and the label are computed as integer lists starting from
    Mapillary filtered images (as a remark, there are no label for the testing
    data set)

    Parameters
    ----------
    datapath: object
        string designing the image dataset relative path
    dataset: object
        string designing the dataset type (either `training` or `validation`)
    size: list
        desired image size (two integers, resp. width and height)
    nb_labels: integer
        number of label in the Mapillary classification

    """
    logger.info("Generating images from {} dataset...".format(dataset))
    IMAGE_PATH = os.path.join(datapath, dataset, "images")
    INPUT_PATH = os.path.join(datapath, dataset, "input")
    make_dir(INPUT_PATH)
    LABEL_PATH = os.path.join(datapath, dataset, "labels")
    OUTPUT_PATH = os.path.join(datapath, dataset, "output")
    make_dir(OUTPUT_PATH)
    train_labels = []
    for img_id, img_filename in enumerate(os.listdir(IMAGE_PATH)):
        img_in = Image.open(os.path.join(IMAGE_PATH, img_filename))
        old_width, old_height = img_in.size
        # pick ten (x, y) couples
        x = np.random.randint(0, old_width-size[0], 10)
        y = np.random.randint(0, old_height-size[1], 10)
        # crop ten sub-images starting from the original one
        for i, x_, y_ in zip(range(len(x)), x, y):
            new_img = img_in.crop((x_, y_, x_+size[0], y_+size[1]))
            if i % 2 == 1:
                new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
            new_img_name = "{:05d}{}.jpg".format(img_id, i)
            instance_name = os.path.join(INPUT_PATH, new_img_name)
            new_img.save(instance_name)
            logger.info(("[{} set] Create image {} from {}..."
                         "").format(dataset, new_img_name, img_filename))
            label_filename = img_filename.replace(".jpg", ".png")
            img_out = Image.open(os.path.join(LABEL_PATH,
                                              label_filename))
            img_out = img_out.crop((x_, y_, x_+size[0], y_+size[1]))
            labels_out = mapillary_label_building(img_out, nb_labels)
            labels_out.insert(0, old_height)
            labels_out.insert(0, old_width)
            labels_out.insert(0, new_img_name)
            labels_out.insert(0, img_filename)
            train_labels.append(labels_out)
    train_label_descr = (["name", "origin", "width", "height"] +
                         ["label"+str(i) for i in range(nb_labels)])
    train_labels = pd.DataFrame(train_labels,
                                columns=train_label_descr)
    train_labels.to_csv(os.path.join(OUTPUT_PATH, "labels.csv"),
                        index=False)
