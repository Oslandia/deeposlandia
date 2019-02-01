"""Dataset modules

Each considered dataset is represented by its own module, and its own class
that inherits from the generic Dataset class.
"""

import abc
import os
import json
import math
from multiprocessing import Pool

import cv2
import daiquiri
import numpy as np
from PIL import Image

from deeposlandia import utils


logger = daiquiri.getLogger(__name__)


AVAILABLE_DATASETS = [
    'shapes', 'mapillary', 'aerial', 'tanzania'
]
GEOGRAPHIC_DATASETS = ["aerial", "tanzania"]

class Dataset(metaclass=abc.ABCMeta):
    """Generic class that describes the behavior of a Dataset object: it is initialized at least
    with an image size, its label are added always through the same manner, it can be serialized (save) and
    deserialized (load) from/to a `.json` file

    Attributes
    ----------
    image_size : int
        Size of considered images (height=width), raw images will be resized during the
    preprocessing
    """
    def __init__(self, image_size):
        if not image_size % 16 == 0:
            raise ValueError("The chosen image size is not divisible "
                             "per 16. To train a neural network with "
                             "such an input size may fail.")
        self.image_size = image_size
        self.label_info = []
        self.image_info = []

    @property
    def label_ids(self):
        """Return the list of labels ids taken into account in the dataset

        They can be grouped.

        Returns
        -------
        list
            List of label ids
        """
        return [label_id for label_id, attr in enumerate(self.label_info)
                if attr['is_evaluate']]

    @property
    def labels(self):
        """Return the description of label that will be evaluated during the process
        """
        return [label for label in self.label_info if label["is_evaluate"]]

    def get_nb_labels(self, see_all=False):
        """Return the number of labels

        Parameters
        ----------
        see_all : boolean
            If True, consider all labels, otherwise consider only labels for which `is_evaluate` is
        True
        """
        if see_all:
            return len(self.label_info)
        else:
            return len(self.label_ids)

    def get_nb_images(self):
        """ `image_info` getter, return the size of `image_info`, i.e. the
        number of images in the dataset
        """
        return len(self.image_info)

    def get_label_popularity(self):
        """Return the label popularity in the current dataset, *i.e.* the proportion of images that
        contain corresponding object
        """
        labels = [img["labels"] for img in self.image_info]
        if self.get_nb_images() == 0:
            logger.error("No images in the dataset.")
            return None
        else:
            return np.round(np.divide(sum(np.array([list(l.values()) for l in labels])),
                                      self.get_nb_images()), 3)


    def add_label(self, label_id, label_name, color, is_evaluate,
                  category=None, aggregated_label_ids=None,
                  contained_labels=None):
        """ Add a new label to the dataset with label id `label_id`

        Parameters
        ----------
        label_id : integer
            Id of the new label
        label_name : str
            String designing the new label name
        color : list
            List of three integers (between 0 and 255) that characterizes the
            label (useful for semantic segmentation result printing)
        is_evaluate : bool
        category : str
            String designing the category of the dataset label
        aggregate_label_ids : list (optional)
            List of label ids aggregated by the current label_id
        contained_labels : list
            List of raw labels aggregated by the current label
        """
        if label_id in self.label_info:
            logger.error("Label %s already stored into the label set."
                         , image_id)
            return None
        category = label_name if category is None else category
        contains = label_name if contained_labels is None else contained_labels
        self.label_info.append({"name": label_name,
                                "id": label_id,
                                "category": category,
                                "is_evaluate": is_evaluate,
                                "aggregate": aggregated_label_ids,
                                "contains": contained_labels,
                                "color": color})

    def save(self, filename):
        """Save dataset in a json file indicated by `filename`

        Parameters
        ----------
        filename : str
            String designing the relative path where the dataset must be saved
        """
        with open(filename, 'w') as fp:
            json.dump({"image_size": self.image_size,
                       "labels": self.label_info,
                       "images": self.image_info}, fp)
        logger.info("The dataset has been saved into %s", filename)

    def load(self, filename, nb_images=None):
        """Load a dataset from a json file indicated by `filename` ; use dict comprehension instead
        of direct assignments in order to convert dict keys to integers

        Parameters
        ----------
        filename : str
            String designing the relative path from where the dataset must be
        loaded
        nb_images : integer
            Number of images that must be loaded (if None, the whole dataset is loaded)
        """
        with open(filename) as fp:
            ds = json.load(fp)
        self.image_size = ds["image_size"]
        self.label_info = ds["labels"]
        if nb_images is None:
            self.image_info = ds["images"]
        else:
            self.image_info = ds["images"][:nb_images]
        logger.info("The dataset has been loaded from %s", filename)

    @abc.abstractmethod
    def populate(self):
        """
        """
        pass
