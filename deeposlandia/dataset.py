"""Encapsulates datasets in a generic Dataset class, and in some more specific classes that inherit
from it

"""

import os
import json
import math
from multiprocessing import Pool

import cv2
from PIL import Image

import numpy as np

from deeposlandia import utils

class Dataset:
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
            utils.logger.error("No images in the dataset.")
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
            utils.logger.error("Label {} already stored into the label set.".format(label_id))
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
        utils.logger.info("The dataset has been saved into {}".format(filename))

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
        utils.logger.info("The dataset has been loaded from {}".format(filename))


class AerialDataset(Dataset):
    """Dataset structure inspired from AerialImageDataset, a dataset released
    by INRIA

    The dataset is freely available at:
    https://project.inria.fr/aerialimagelabeling/files/

    It is composed of 180 training images and 180 testing images of size
    5000*5000. There are ground-truth labels only for the former set.

    Attributes
    ----------
    tile_size : int
        Size of the tiles into which each raw images is decomposed during
    dataset population (height=width)

    """

    def __init__(self, tile_size):
        """ Class constructor ; instanciates a AerialDataset as a standard
        Dataset which is completed by a glossary file that describes the
        dataset labels and images

        """
        self.tile_size = tile_size
        img_size = utils.get_image_size_from_tile(self.tile_size)
        super().__init__(img_size)
        self.add_label(label_id=0, label_name="background",
                       color=0, is_evaluate=True)
        self.add_label(label_id=1, label_name="building",
                       color=255, is_evaluate=True)

    def _preprocess(self, image_filename, output_dir, labelling):
        """Resize/crop then save the training & label images

        Parameters
        ----------
        image_filename : str
            Full path towards the image on the disk
        datadir : str
            Output path where preprocessed image must be saved

        Returns
        -------
        dict
            Key/values with the filenames and label ids
        """
        img_in = Image.open(image_filename)
        raw_img_size = img_in.size[0]
        result_dicts = []
        # crop tile_size*tile_size tiles into 5000*5000 raw images
        buffer_tiles = []
        for x in range(0, raw_img_size, self.tile_size):
            for y in range(0, raw_img_size, self.tile_size):
                tile = img_in.crop((x, y,
                                    x + self.tile_size, y + self.tile_size))
                tile = utils.resize_image(tile, self.image_size)
                img_id = int((raw_img_size / self.tile_size
                              * x / self.tile_size
                              + y / self.tile_size))
                basename_decomp = os.path.splitext(
                    os.path.basename(image_filename))
                new_in_filename = (basename_decomp[0] + '_' +
                                   str(img_id) + basename_decomp[1])
                new_in_path = os.path.join(output_dir, 'images', new_in_filename)
                tile.save(new_in_path)
                result_dicts.append({"raw_filename": image_filename,
                                     "image_filename": new_in_path})

        if labelling:
            label_filename = image_filename.replace("images/", "gt/")
            img_out = Image.open(label_filename) 
            buffer_tiles = []
            for x in range(0, raw_img_size, self.tile_size):
                for y in range(0, raw_img_size, self.tile_size):
                    tile = img_out.crop((x, y,
                                         x + self.tile_size, y + self.tile_size))
                    tile = utils.resize_image(tile, self.image_size)
                    img_id = int((raw_img_size / self.tile_size
                                  * x / self.tile_size
                                  + y / self.tile_size))
                    basename_decomp = os.path.splitext(
                        os.path.basename(image_filename))
                    new_out_filename = (basename_decomp[0] + '_' +
                                       str(img_id) + basename_decomp[1])
                    new_out_path = os.path.join(output_dir, 'labels',
                                                new_out_filename)
                    tile.save(new_out_path)
                    labels = utils.label_building(tile,
                                                  self.label_ids,
                                                  dataset='aerial')
                    result_dicts[img_id]["label_filename"] = new_out_path
                    result_dicts[img_id]["labels"] = labels

        return result_dicts

    def populate(self, output_dir, input_dir, nb_images=None,
                 aggregate=False, labelling=True):
        """ Populate the dataset with images contained into `datadir` directory

        Parameters
        ----------
        output_dir : str
            Path of the directory where the preprocessed image must be saved
        input_dir : str
            Path of the directory that contains input images
        nb_images : integer
            Number of images to be considered in the dataset; if None, consider the whole
        repository
        aggregate : bool
            Label aggregation parameter, useless for this dataset, but kept for
        class method genericity
        labelling: boolean
            If True labels are recovered from dataset, otherwise dummy label are generated
        """
        image_list = os.listdir(os.path.join(input_dir, "images"))
        image_list_longname = [os.path.join(input_dir, "images", l)
                               for l in image_list if not l.startswith('.')][:nb_images]
        utils.logger.info(("Getting {} images to preprocess..."
                           "").format(len(image_list_longname)))
        with Pool() as p:
            self.image_info = p.starmap(self._preprocess,
                                        [(x, output_dir, labelling)
                                         for x in image_list_longname])
        self.image_info = [item for sublist in self.image_info
                           for item in sublist]
        utils.logger.info(("Saved {} images in the preprocessed dataset."
                           "").format(len(self.image_info)))


class MapillaryDataset(Dataset):
    """Dataset structure that gathers all information related to the Mapillary images

    Attributes
    ----------
    image_size : int
        Size of considered images (height=width), raw images will be resized during the
    preprocessing
    glossary_filename : str
        Name of the Mapillary input glossary, that contains every information about Mapillary
    labels

    """

    def __init__(self, image_size, glossary_filename):
        """ Class constructor ; instanciates a MapillaryDataset as a standard Dataset which is
        completed by a glossary file that describes the dataset labels
        """
        super().__init__(image_size)
        self.build_glossary(glossary_filename)

    def build_glossary(self, config_filename):
        """Read the Mapillary glossary stored as a json file at the data
        repository root

        Parameters
        ----------
        config_filename : str
            String designing the relative path of the dataset glossary
        (based on Mapillary dataset)
        """
        with open(config_filename) as config_file:
            glossary = json.load(config_file)
        if "labels" not in glossary:
            utils.logger.error("There is no 'label' key in the provided glossary.")
            return None
        for lab_id, label in enumerate(glossary["labels"]):
            name_items = label["name"].split('--')
            category = '-'.join(name_items)
            self.add_label(lab_id, name_items, label["color"],
                           label['evaluate'], category, label["contains_id"],
                           label['contains'])

    def group_image_label(self, image):
        """Group the labels

        If the label ids 4, 5 and 6 belong to the same group, they will be turned
        into the label id 4.

        Parameters
        ----------
        image : PIL.Image

        Returns
        -------
        PIL.Image
        """
        # turn all label ids into the lowest digits/label id according to its "group"
        # (manually built)
        a = np.array(image)
        for root_id, label in enumerate(self.label_info):
            for label_id in label['aggregate']:
                mask = a == label_id
                a[mask] = root_id
        return Image.fromarray(a, mode=image.mode)

    def _preprocess(self, image_filename, output_dir, aggregate, labelling=True):
        """Resize/crop then save the training & label images

        Parameters
        ----------
        datadir : str
        image_filaname : str
        aggregate : boolean
        labelling : boolean

        Returns
        -------
        dict
            Key/values with the filenames and label ids
        """
        # open original images
        img_in = Image.open(image_filename)

        # resize images (self.image_size*larger_size or larger_size*self.image_size)
        img_in = utils.resize_image(img_in, self.image_size)

        # crop images to get self.image_size*self.image_size dimensions
        crop_pix = np.random.randint(0, 1 + max(img_in.size) - self.image_size)
        final_img_in = utils.mono_crop_image(img_in, crop_pix)

        # save final image
        new_in_filename = os.path.join(output_dir, 'images',
                                       os.path.basename(image_filename))
        final_img_in.save(new_in_filename)

        # label_filename vs label image
        if labelling:
            label_filename = image_filename.replace("images/", "labels/")
            label_filename = label_filename.replace(".jpg", ".png")
            img_out = Image.open(label_filename)
            img_out = utils.resize_image(img_out, self.image_size)
            final_img_out = utils.mono_crop_image(img_out, crop_pix)
            # group some labels
            if aggregate:
                final_img_out = self.group_image_label(final_img_out)

            labels = utils.label_building(final_img_out,
                                          self.label_ids,
                                          dataset="mapillary")
            new_out_filename = os.path.join(output_dir, 'labels',
                                            os.path.basename(label_filename))
            final_img_out = utils.build_image_from_config(final_img_out,
                                                          self.label_info)
            final_img_out.save(new_out_filename)
        else:
            new_out_filename = None
            labels = {i: 0 for i in range(self.get_nb_labels())}

        return {"raw_filename": image_filename,
                "image_filename": new_in_filename,
                "label_filename": new_out_filename,
                "labels": labels}

    def populate(self, output_dir, input_dir, nb_images=None, aggregate=False, labelling=True):
        """ Populate the dataset with images contained into `datadir` directory

        Parameters
        ----------
        output_dir : str
            Path of the directory where the preprocessed image must be saved
        input_dir : str
            Path of the directory that contains input images
        nb_images : integer
            Number of images to be considered in the dataset; if None, consider the whole
        repository
        aggregate : bool
            Aggregate some labels into more generic ones, e.g. cars and bus into the vehicle label
        labelling: boolean
            If True labels are recovered from dataset, otherwise dummy label are generated
        """
        image_list = os.listdir(os.path.join(input_dir, "images"))[:nb_images]
        image_list_longname = [os.path.join(input_dir, "images", l) for l in image_list]
        with Pool() as p:
            self.image_info = p.starmap(self._preprocess, [(x, output_dir, aggregate, labelling)
                                                  for x in image_list_longname])


class ShapeDataset(Dataset):
    """Dataset structure that gathers all information related to a randomly-generated shape Dataset

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
        Size of considered images (height=width), raw images will be resized during the
    preprocessing
    nb_labels : int
        Number of shape types that must be integrated into the dataset (only 1, 2 and 3 are supported)

    """

    SQUARE = 0
    SQUARE_COLOR = (50, 50, 200) # Blue
    CIRCLE = 1
    CIRCLE_COLOR = (200, 50, 50) # Red
    TRIANGLE = 2
    TRIANGLE_COLOR = (50, 200, 50) # Green
    BACKGROUND = 3
    BACKGROUND_COLOR = (200, 200, 200) # Light grey

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
        self.add_label(self.BACKGROUND, "background", self.BACKGROUND_COLOR, True)

    def generate_labels(self, nb_images):
        """ Generate random shape labels in order to prepare shape image
        generation; use numpy to generate random indices for each labels, these
        indices will be the positive examples; return a 2D-list

        Parameters
        ----------
        nb_images : integer
            Number of images to label in the dataset
        """
        raw_labels = [np.random.choice(np.arange(nb_images),
                                            int(nb_images/2),
                                            replace=False)
                      for i in range(self.get_nb_labels())]
        labels = np.zeros([nb_images, self.get_nb_labels()], dtype=int)
        for i in range(self.get_nb_labels()):
            labels[raw_labels[i], i] = 1
        return [dict([(i, int(j)) for i, j in enumerate(l)]) for l in labels]

    def populate(self, output_dir=None, input_dir=None, nb_images=10000, aggregate=False, labelling=True, buf=8):
        """ Populate the dataset with images contained into `datadir` directory

        Parameters
        ----------
        output_dir : str
            Path of the directory where the preprocessed image must be saved
        input_dir : str
            Path of the directory that contains input images
        nb_images: integer
            Number of images that must be added in the dataset
        aggregate: bool
            Aggregate some labels into more generic ones, e.g. cars and bus into the vehicle label
        labelling: boolean
            Dummy parameter: in this dataset, labels are always generated, as images are drawed with them
        buf: integer
            Minimal number of pixels between shape base point and image borders
        """
        shape_gen = self.generate_labels(nb_images)
        for i, image_label in enumerate(shape_gen):
            bg_color = np.random.randint(0, 255, 3).tolist()
            shape_specs = []
            for l in image_label.items():
                if l:
                    shape_color = np.random.randint(0, 255, 3).tolist()
                    x, y = np.random.randint(buf, self.image_size - buf - 1, 2).tolist()
                    shape_size = np.random.randint(buf, self.image_size // 4)
                    shape_specs.append([shape_color, x, y, shape_size])
                else:
                    shape_specs.append([None, None, None, None])
            self.add_image(i, bg_color, shape_specs, image_label)
            if not output_dir is None:
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
            utils.logger.error("Image {} already stored into the label set.".format(image_id))
            return None
        self.image_info.append({"background": background,
                                "shape_specs": specifications,
                                "labels": labels})

    def draw_image(self, image_id, datapath):
        """Draws an image from the specifications of its shapes and saves it on
        the file system to `datapath`

        Save labels as mono-channel images on the file system by using the label ids

        Parameters
        ----------
        image_id : integer
            Image id
        datapath : str
            String that characterizes the repository in which images will be stored
        """
        image_info = self.image_info[image_id]

        image = np.ones([self.image_size, self.image_size, 3], dtype=np.uint8)
        image = image * np.array(image_info["background"], dtype=np.uint8)
        label = np.full([self.image_size, self.image_size, 3], self.BACKGROUND_COLOR, dtype=np.uint8)

        # Get the center x, y and the size s
        if image_info["labels"][self.SQUARE]:
            color, x, y, s = image_info["shape_specs"][self.SQUARE]
            color = tuple(map(int, color))
            image = cv2.rectangle(image, (x - s, y - s), (x + s, y + s), color, -1)
            label = cv2.rectangle(label, (x - s, y - s), (x + s, y + s), self.SQUARE_COLOR, -1)
        if image_info["labels"][self.CIRCLE]:
            color, x, y, s = image_info["shape_specs"][self.CIRCLE]
            color = tuple(map(int, color))
            image = cv2.circle(image, (x, y), s, color, -1)
            label = cv2.circle(label, (x, y), s, self.CIRCLE_COLOR, -1)
        if image_info["labels"][self.TRIANGLE]:
            color, x, y, s = image_info["shape_specs"][self.TRIANGLE]
            color = tuple(map(int, color))
            x, y, s = map(int, (x, y, s))
            points = np.array([[(x, y - s),
                                (x - s / math.sin(math.radians(60)), y + s),
                                (x + s / math.sin(math.radians(60)), y + s),]],
                              dtype=np.int32)
            image = cv2.fillPoly(image, points, color)
            label = cv2.fillPoly(label, points, self.TRIANGLE_COLOR)
        image_filename = os.path.join(datapath, "images", "shape_{:05}.png".format(image_id))
        self.image_info[image_id]["image_filename"] = image_filename
        Image.fromarray(image).save(image_filename)
        label_filename = os.path.join(datapath, "labels", "shape_{:05}.png".format(image_id))
        self.image_info[image_id]["label_filename"] = label_filename
        Image.fromarray(label).save(label_filename)
