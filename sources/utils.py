# Author: Raphael Delhome
# Organization: Oslandia
# Date: september 2017

# Utilitary function for Mapillary dataset analysis

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import sys

DATASET = ["training", "validation", "testing"]
IMG_SIZE = (768, 576) # easy decomposition: (4, 3) * 3 * 2 * 2 * 2 * 2 * 2 * 2
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
    return [min(math.log(mu * nb_images / l), max_weight) for l in label_counter]

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

def are_labels_equal(old_label, new_label):
    """Verify if two lists contain the same values, at the same location;
    these lists must contain comparable items

    Parameters
    ----------
    old_label: list
        First list to compare
    new_label: list
        Second list to compare
    
    """
    return sum(new_label == old_label) == len(old_label)

def check_label_equality(dataset, labels, img_id):
    """Verify if the raw filtered images and the new ones (generated after
    resizing operations) are equally labeled, and return the index of the
    issuing label(s) when it is not the case

    Parameters
    ----------
    dataset: object
        string designing the considered repository (`training`, `validation` or `testing`)
    labels: pd.DataFrame
        table that contains the metadata information about each filtered images
    (ex: their name on the file system)
    img_id: integer
        index of the checked image
    
    """
    image = Image.open(os.path.join("..", "data", dataset, "labels",
                                    labels.iloc[img_id, 0].replace(".jpg",
                                                                   ".png")))
    old_label = np.array(mapillary_label_building(image, 66))
    new_label = labels.iloc[img_id, 6:]
    return {"invalid_label": np.where(new_label != old_label)[0],
            "pixel_count": (pd.Series(np.array(image).reshape([-1]))
                            .value_counts())}

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

def mapillary_data_preparation(dataset, nb_labels):
    """Prepare the Mapillary dataset for processing convolutional neural network
    computing: the images are resized and renamed as `index.jpg`, where `index`
    is an integer comprised between 0 and 17999 (size of the validation
    dataset), and the label are computed as integer lists starting from
    Mapillary filtered images (as a remark, there are no label for the testing
    data set)

    Parameters
    ----------
    dataset: object
        string designing the image data (`training`, `validation` or `testing`)
    nb_labels: integer
        number of label in the Mapillary classification

    """
    IMAGE_PATH = os.path.join("..", "data", dataset, "images")
    INPUT_PATH = os.path.join("..", "data", dataset, "input")
    make_dir(INPUT_PATH)
    if dataset != "testing":
        LABEL_PATH = os.path.join("..", "data", dataset, "labels")
        OUTPUT_PATH = os.path.join("..", "data", dataset, "output")
        make_dir(OUTPUT_PATH)
        train_y = []
    for img_id, img_filename in enumerate(os.listdir(IMAGE_PATH)):
        img_in = Image.open(os.path.join(IMAGE_PATH, img_filename))
        old_width, old_height = img_in.size
        img_in = img_in.resize(IMG_SIZE, Image.NEAREST)
        new_img_name = "{:05d}.jpg".format(img_id)
        instance_name = os.path.join(INPUT_PATH, new_img_name)
        img_in.save(instance_name)
        logger.info("""[{} set] Image {} saved as {}..."""
                       .format(dataset, img_filename, new_img_name))
        if dataset != "testing":
            label_filename = img_filename.replace(".jpg", ".png")
            img_out = Image.open(os.path.join(LABEL_PATH, label_filename))
            img_out = img_out.resize(IMG_SIZE, Image.NEAREST)
            y = mapillary_label_building(img_out, nb_labels)
            width_ratio = IMG_SIZE[0] / old_width
            height_ratio = IMG_SIZE[1] / old_height
            y.insert(0, height_ratio)
            y.insert(0, old_height)
            y.insert(0, width_ratio)
            y.insert(0, old_width)
            y.insert(0, new_img_name)
            y.insert(0, img_filename)
            train_y.append(y)
    if dataset != "testing":
        train_y = pd.DataFrame(train_y, columns=["old_name", "new_name",
                                                 "old_width", "width_ratio",
                                                 "old_height", "height_ratio"]
                               + ["label_" + str(i) for i in range(nb_labels)])
        train_y.to_csv(os.path.join(OUTPUT_PATH, "labels.csv"), index=False)

def mapillary_output_checking(dataset, nb_labels):
    """Verify if every images are well-labeled, i.e. that new image names
    correspond to old ones, and that resizing operations did not affect output
    labels too much; images are not the same in the first case, thus labels
    should be largely different; in the second case we can expect some minor
    label modification (low-frequency classes, i.e. with just a few pixels in
    the raw images, may have disappeared in new image versions)

    This function only provides logs that describe the image alterations, it
    does not break the code.
    
    Parameters
    ----------
    dataset: object
        string designing the considered dataset (`training`, `validation` or
    `testing`)
    nb_labels: integer
        number of labels in the Mapillary classification
    
    """
    LABEL_PATH = os.path.join("..", "data", dataset, "labels")
    OUTPUT_PATH = os.path.join("..", "data", dataset, "output")
    new_labels = pd.read_csv(os.path.join(OUTPUT_PATH, "labels.csv"))
    for new_filename in new_labels.new_name:
        current_label = new_labels.query("new_name == @new_filename")
        new_label = np.array(current_label.iloc[0,6:])
        label_filename = current_label['old_name'].values[0]
        img_filename = label_filename.replace(".jpg", ".png")
        img_out = Image.open(os.path.join(LABEL_PATH, img_filename))
        old_label = mapillary_label_building(img_out, nb_labels)
        equality = are_labels_equal(old_label, new_label)
        if equality:
            logger.info("""[{} set] Image {} OK""".format(dataset, new_filename))
        else:
            logger.info("""[{} set] Image {} not OK:\
            {}""".format(dataset, new_filename, current_label.iloc[0,:6]))
            img_id = int(current_label['new_name'].values[0].split('.')[0])
            logger.info(check_label_equality(dataset, new_labels, img_id))
