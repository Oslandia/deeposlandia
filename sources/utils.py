# Author: Raphael Delhome, Oslandia

# Utilitary function for Mapillary dataset analysis

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
logger.addHandler(ch)

DATASET = ["training", "validation", "testing"]
IMG_SIZE = (768, 576) # easy decomposition: (4, 3) * 3 * 2 * 2 * 2 * 2 * 2 * 2
IMAGE_TYPES = ["images", "instances", "labels"]
TRAINING_IMAGE_PATH = os.path.join("data", "training", "images")
TRAINING_LABEL_PATH = os.path.join("data", "training", "labels")
TRAINING_INPUT_PATH = os.path.join("data", "training", "input",
                                   "{}_{}".format(IMG_SIZE[0], IMG_SIZE[1]))
TRAINING_OUTPUT_PATH = os.path.join("data", "training", "output")
VALIDATION_IMAGE_PATH = os.path.join("data", "validation", "images")
VALIDATION_LABEL_PATH = os.path.join("data", "validation", "labels")
VALIDATION_INPUT_PATH = os.path.join("data", "validation", "input",
                                   "{}_{}".format(IMG_SIZE[0], IMG_SIZE[1]))
VALIDATION_OUTPUT_PATH = os.path.join("data", "validation", "output")

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def unnest(l):
    """Unnest a list of lists, by splitting sublists and returning a simple list of scalar elements

    Parameters
    ----------
    l: list
        list of lists
    
    """
    return [index for sublist in l for index in sublist]
    
def mapillary_label_building(filtered_image, nb_labels):
    filtered_data = np.array(filtered_image)
    avlble_labels = (pd.Series(filtered_data.reshape([-1]))
                     .value_counts()
                     .index)
    return [1 if i in avlble_labels else 0 for i in range(nb_labels)]

def mapillary_image_size_plot(data, filename):
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
    return sum(new_label == old_label) == len(old_label)

def check_label_equality(dataset, labels, img_id):
    image = Image.open(os.path.join("data", dataset, "labels",
                                    labels.iloc[img_id, 0].replace(".jpg",
                                                                   ".png")))
    old_label = np.array(mapillary_label_building(image, 66))
    new_label = labels.iloc[img_id, 6:]
    return {"invalid_label": np.where(new_label != old_label)[0],
            "pixel_count": pd.Series(np.array(image).reshape([-1])).value_counts()}

def size_inventory():
    sizes = []
    widths = []
    heights = []
    types = []
    filenames = []
    datasets = []
    for dataset in DATASET:
        logger.info("Size inventory in {} dataset".format(dataset))
        for img_filename in os.listdir(os.path.join("data", dataset, "images")):
            for img_type in IMAGE_TYPES:
                if dataset == "testing" and not img_type == "images":
                    continue
                complete_filename = os.path.join("data", dataset,
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

def mapillary_data_preparation(dataset="training", nb_labels=1):
    IMAGE_PATH = os.path.join("data", dataset, "images")
    INPUT_PATH = os.path.join("data", dataset, "input")
    make_dir(INPUT_PATH)
    if dataset != "testing":
        LABEL_PATH = os.path.join("data", dataset, "labels")
        OUTPUT_PATH = os.path.join("data", dataset, "output")
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

def mapillary_output_checking(dataset="training", nb_labels=1):
    LABEL_PATH = os.path.join("data", dataset, "labels")
    OUTPUT_PATH = os.path.join("data", dataset, "output")
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
