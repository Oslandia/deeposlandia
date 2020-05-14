"""Shape dataset module

Model randomly generated images with geometric shapes, in order to get a toy
dataset that will make the convolutional neural network tests easier.
"""

import os
import math

import cv2
import daiquiri
import numpy as np
from PIL import Image

from deeposlandia.datasets import Dataset


logger = daiquiri.getLogger(__name__)


class ShapeDataset(Dataset):
    """Dataset structure that gathers all information related to a
    randomly-generated shape Dataset

    In such a dataset, a set of images is generated with either a square, or a
    circle or a triangle, or two of them, or all of them. A random background
    color is applied, and shape color itself is also randomly generated. Each
    of these labels are characterized with a fixed color for comparison between
    ground truth and predictions: squares, circles and triangles will be
    respectively set as blue, red and green, whilst background will be set as
    light grey.

    Attributes
    ----------
    image_size : int
        Size of considered images (height=width), raw images will be resized
    during the preprocessing
    nb_labels : int
        Number of shape types that must be integrated into the dataset (only 1,
    2 and 3 are supported)

    """

    SQUARE = 0
    SQUARE_COLOR = (50, 50, 200)  # Blue
    CIRCLE = 1
    CIRCLE_COLOR = (200, 50, 50)  # Red
    TRIANGLE = 2
    TRIANGLE_COLOR = (50, 200, 50)  # Green
    BACKGROUND = 3
    BACKGROUND_COLOR = (200, 200, 200)  # Light grey

    def __init__(self, image_size):
        """ Class constructor
        """
        super().__init__(image_size)
        self.build_glossary()

    def build_glossary(self):
        """Read the shape glossary stored as a json file at the data
        repository root

        Parameters
        ----------
        nb_labels : integer
            Number of shape types (either 1, 2 or 3, warning if more)
        """
        self.add_label(self.SQUARE, "square", self.SQUARE_COLOR, True)
        self.add_label(self.CIRCLE, "circle", self.CIRCLE_COLOR, True)
        self.add_label(self.TRIANGLE, "triangle", self.TRIANGLE_COLOR, True)
        self.add_label(
            self.BACKGROUND, "background", self.BACKGROUND_COLOR, True
        )

    def generate_labels(self, nb_images):
        """ Generate random shape labels in order to prepare shape image
        generation; use numpy to generate random indices for each labels, these
        indices will be the positive examples; return a 2D-list

        Parameters
        ----------
        nb_images : integer
            Number of images to label in the dataset
        """
        raw_labels = [
            np.random.choice(
                np.arange(nb_images), int(nb_images / 2), replace=False
            )
            for i in range(self.get_nb_labels())
        ]
        labels = np.zeros([nb_images, self.get_nb_labels()], dtype=int)
        for i in range(self.get_nb_labels()):
            labels[raw_labels[i], i] = 1
        return [dict([(i, int(j)) for i, j in enumerate(l)]) for l in labels]

    def populate(
        self,
        output_dir=None,
        input_dir=None,
        nb_images=10000,
        nb_tiles_per_image=None,
        aggregate=False,
        labelling=True,
        buf=8,
        nb_processes=None
    ):
        """ Populate the dataset with images contained into `datadir` directory

        Parameters
        ----------
        output_dir : str
            Path of the directory where the preprocessed image must be saved
        input_dir : str
            Path of the directory that contains input images
        nb_images: integer
            Number of images that must be added in the dataset
        nb_tiles_per_image : integer
            Number of tiles that must be picked into the raw image (useless there, added
        for consistency)
        aggregate: bool
            Aggregate some labels into more generic ones, e.g. cars and bus
        into the vehicle label
        labelling: boolean
            Dummy parameter: in this dataset, labels are always generated, as
        images are drawed with them
        buf: integer
            Minimal number of pixels between shape base point and image borders
        nb_processes : int
            Number of processes on which to run the preprocessing (dummy
        parameter for "shapes" datasets)
        """
        if nb_tiles_per_image is not None:
            logger.warning("The ``nb_tiles_per_image`` parameter is useless, it will be ignored.")
        shape_gen = self.generate_labels(nb_images)
        for i, image_label in enumerate(shape_gen):
            bg_color = np.random.randint(0, 255, 3).tolist()
            shape_specs = []
            for l in image_label.items():
                if l:
                    shape_color = np.random.randint(0, 255, 3).tolist()
                    x, y = np.random.randint(
                        buf, self.image_size - buf - 1, 2
                    ).tolist()
                    shape_size = np.random.randint(buf, self.image_size // 4)
                    shape_specs.append([shape_color, x, y, shape_size])
                else:
                    shape_specs.append([None, None, None, None])
            self.add_image(i, bg_color, shape_specs, image_label)
            if output_dir is not None:
                self.draw_image(i, output_dir)

    def add_image(self, image_id, background, specifications, labels):
        """ Add a new image to the dataset with image id `image_id`; an image
        in the dataset is represented by an id, a list of shape specifications,
        a background color and a list of 0-1 labels (1 if the i-th label is on
        the image, 0 otherwise)

        Parameters
        ----------
        image_id : integer
            Id of the new image
        background : list
            List of three integer between 0 and 255 that designs the image
        background color
        specifications : list
            Image specifications, as a list of shapes (color, coordinates and
        size)
        labels : list
            List of 0-1 values, the i-th value being 1 if the i-th label is on
        the new image, 0 otherwise; the label list length correspond to the
        number of labels in the dataset
        """
        if image_id in self.image_info:
            logger.error(
                "Image %s already stored into the label set.", image_id
            )
            return None
        self.image_info.append(
            {
                "background": background,
                "shape_specs": specifications,
                "labels": labels,
            }
        )

    def draw_image(self, image_id, datapath):
        """Draws an image from the specifications of its shapes and saves it on
        the file system to `datapath`

        Save labels as mono-channel images on the file system by using the
        label ids

        Parameters
        ----------
        image_id : integer
            Image id
        datapath : str
            String that characterizes the repository in which images will be
        stored
        """
        image_info = self.image_info[image_id]

        image = np.ones([self.image_size, self.image_size, 3], dtype=np.uint8)
        image = image * np.array(image_info["background"], dtype=np.uint8)
        label = np.full(
            [self.image_size, self.image_size, 3],
            self.BACKGROUND_COLOR,
            dtype=np.uint8,
        )

        # Get the center x, y and the size s
        if image_info["labels"][self.SQUARE]:
            color, x, y, s = image_info["shape_specs"][self.SQUARE]
            color = tuple(map(int, color))
            image = cv2.rectangle(
                image, (x - s, y - s), (x + s, y + s), color, -1
            )
            label = cv2.rectangle(
                label, (x - s, y - s), (x + s, y + s), self.SQUARE_COLOR, -1
            )
        if image_info["labels"][self.CIRCLE]:
            color, x, y, s = image_info["shape_specs"][self.CIRCLE]
            color = tuple(map(int, color))
            image = cv2.circle(image, (x, y), s, color, -1)
            label = cv2.circle(label, (x, y), s, self.CIRCLE_COLOR, -1)
        if image_info["labels"][self.TRIANGLE]:
            color, x, y, s = image_info["shape_specs"][self.TRIANGLE]
            color = tuple(map(int, color))
            x, y, s = map(int, (x, y, s))
            points = np.array(
                [
                    [
                        (x, y - s),
                        (x - s / math.sin(math.radians(60)), y + s),
                        (x + s / math.sin(math.radians(60)), y + s),
                    ]
                ],
                dtype=np.int32,
            )
            image = cv2.fillPoly(image, points, color)
            label = cv2.fillPoly(label, points, self.TRIANGLE_COLOR)
        image_filename = os.path.join(
            datapath, "images", "shape_{:05}.png".format(image_id)
        )
        self.image_info[image_id]["image_filename"] = image_filename
        Image.fromarray(image).save(image_filename)
        label_filename = os.path.join(
            datapath, "labels", "shape_{:05}.png".format(image_id)
        )
        self.image_info[image_id]["label_filename"] = label_filename
        Image.fromarray(label).save(label_filename)
