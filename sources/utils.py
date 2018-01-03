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

def one_hot_encoding(image_filename, nb_labels):
    """Build a list of integer labels that are contained into a candidate
    filtered image designed by its name on file system; the labels are
    recognized starting from image pixels

    Parameters
    ----------
    image_filename: object
        File name of the image that has to be encoded
    nb_labels: integer
        number of labels contained into the reference classification

    """
    image = Image.open(image_filename)
    return mapillary_label_building(image, nb_labels)

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
    image_data = np.array(filtered_image)
    available_labels = np.unique(image_data)
    return [1 if i in available_labels else 0 for i in range(nb_labels)]

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
    size: integer
        desired image size (width and height are equals)
    nb_labels: integer
        number of label in the Mapillary classification

    """
    logger.info("Generating images from {} dataset...".format(dataset))
    IMAGE_PATH = os.path.join(datapath, dataset, "images")
    INPUT_PATH = os.path.join(datapath, dataset, "input_" + str(size))
    LABEL_PATH = os.path.join(datapath, dataset, "labels")
    OUTPUT_PATH = os.path.join(datapath, dataset, "output_" + str(size))
    make_dir(INPUT_PATH)
    make_dir(OUTPUT_PATH)
    image_files = os.listdir(IMAGE_PATH)
    train_label_descr = (["name", "raw_image",
                          "old_width", "old_height", "pixel_ratio"] +
                         ["label"+str(i) for i in range(nb_labels)])
    train_labels = [pd.DataFrame(generate_images(img_id, img_fn, size, 10, nb_labels,
                                                 IMAGE_PATH, INPUT_PATH,
                                                 LABEL_PATH), columns=train_label_descr)
                    for img_id, img_fn in enumerate(image_files)]
    train_labels = pd.concat(train_labels)
    train_labels = train_labels.sort_values(by="name")
    train_labels.to_csv(os.path.join(OUTPUT_PATH, "labels.csv"),
                        index=False)

def generate_images(img_id, img_filename, base_size, nb_subimages, nb_labels,
                    image_path, input_path, label_path):
    """Generate a bunch of sub-images from another image by cropping operations

    Parameter
    ---------
    img_id: integer
        Raw image id, that will be integrated in the new image names
    img_filename: object
        String designing the name of the starting image
    base_size: integer
        Desired image size (width=height)
    nb_sumimages: integer
        Number of generated sub-images from each input image
    nb_labels: integer
        Total number of labels within the glossary
    image_path: object
        String designing the relative path to images
    input_path: object
        String designing the relative path in which new images must be saved
    label_path: object
        String designing the relative path to labelled images
    """

    logger.info(("Create images {} to {} from {}..."
                 "").format(img_id*10, (img_id+1)*10-1, img_filename))
    img_in = Image.open(os.path.join(image_path, img_filename))
    img_out = Image.open(os.path.join(label_path,
                                      img_filename.replace(".jpg", ".png")))
    # Resizing
    old_width, old_height = img_in.size
    img_in = resize_image(img_in, base_size*2)
    img_out = resize_image(img_out, base_size*2)
    new_width, new_height = img_in.size
    resizing_ratio = math.ceil(old_width * old_height / (new_width * new_height))
    # Sub-image generation
    labels = []
    x = np.random.randint(0, img_in.size[0] - base_size, nb_subimages)
    y = np.random.randint(0, img_in.size[1] - base_size, nb_subimages)
    for i, x_, y_ in zip(range(len(x)), x, y):
        new_img = crop_image(img_in, (x_, y_, x_ + base_size, y_ + base_size))
        new_img = flip_image(new_img)
        new_img_name = "{:05d}{}.jpg".format(img_id, i)
        new_img.save(os.path.join(input_path, new_img_name))
        labels_out = [new_img_name, img_filename, img_in.size[0],
                      img_in.size[1], resizing_ratio]
        new_img_out = img_out.crop((x_, y_, x_ + base_size, y_ + base_size))
        labels_out = labels_out + mapillary_label_building(new_img_out, nb_labels)
        labels.append(labels_out)
    return labels

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

def crop_image(img, coordinates):
    """ Crop image `img` following coordinates `coordinates`; simple overload
    of the Image.crop() procedure

    Parameters:
    -----------
    img: Image
        input image to resize
    coordinates: list/tuple
        iterable object of 4 elements (left, up, right, down) that bound the
    cropping process
    """
    return img.crop(coordinates)

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

def split_list(l, nb_partitions):
    """Split a list `l` into `nb_partitions` equal sublists

    Parameters
    ----------
    l: list
        List of elements to split
    nb_partitions: integer
        number of desired partitions
    """
    return [l[i:(i+len(l)/nb_partitions)]
            for i in range(0, len(l), int(len(l)/nb_partitions))]
