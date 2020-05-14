"""Dataset modules

Each considered dataset is represented by its own module, and its own class
that inherits from the generic Dataset class.
"""

import abc
import json
from multiprocessing import Pool
import os

import cv2
import daiquiri
import geopandas as gpd
import numpy as np
from osgeo import gdal
from PIL import Image

from deeposlandia import geometries

logger = daiquiri.getLogger(__name__)


AVAILABLE_DATASETS = ("shapes", "mapillary", "aerial", "tanzania")
GEOGRAPHIC_DATASETS = ("aerial", "tanzania")


class Dataset(metaclass=abc.ABCMeta):
    """Generic class that describes the behavior of a Dataset object: it is
    initialized at least with an image size, its label are added always through
    the same manner, it can be serialized (save) and deserialized (load)
    from/to a `.json` file

    Attributes
    ----------
    image_size : int
        Size of considered images (height=width), raw images will be resized
    during the preprocessing
    """

    def __init__(self, image_size):
        if not image_size % 16 == 0:
            raise ValueError(
                "The chosen image size is not divisible "
                "per 16. To train a neural network with "
                "such an input size may fail."
            )
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
        return [
            label_id
            for label_id, attr in enumerate(self.label_info)
            if attr["is_evaluate"]
        ]

    @property
    def labels(self):
        """Return the description of label that will be evaluated during the
    process
        """
        return [label for label in self.label_info if label["is_evaluate"]]

    def get_nb_labels(self, see_all=False):
        """Return the number of labels

        Parameters
        ----------
        see_all : boolean
            If True, consider all labels, otherwise consider only labels for
        which `is_evaluate` is True
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
        """Return the label popularity in the current dataset, *i.e.* the
        proportion of images that contain corresponding object
        """
        labels = [img["labels"] for img in self.image_info]
        if self.get_nb_images() == 0:
            logger.error("No images in the dataset.")
            return None
        else:
            return np.round(
                np.divide(
                    sum(np.array([list(l.values()) for l in labels])),
                    self.get_nb_images(),
                ),
                3,
            )

    def add_label(
        self,
        label_id,
        label_name,
        color,
        is_evaluate,
        category=None,
        aggregated_label_ids=None,
        contained_labels=None,
    ):
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
            logger.error(
                "Label %s already stored into the label set.", label_id
            )
            return None
        category = label_name if category is None else category
        contains = label_name if contained_labels is None else contained_labels
        self.label_info.append(
            {
                "name": label_name,
                "id": label_id,
                "category": category,
                "is_evaluate": is_evaluate,
                "aggregate": aggregated_label_ids,
                "contains": contains,
                "color": color,
            }
        )

    def save(self, filename):
        """Save dataset in a json file indicated by `filename`

        Parameters
        ----------
        filename : str
            String designing the relative path where the dataset must be saved
        """
        with open(filename, "w") as fp:
            json.dump(
                {
                    "image_size": self.image_size,
                    "labels": self.label_info,
                    "images": self.image_info,
                },
                fp,
            )
        logger.info("The dataset has been saved into %s", filename)

    def load(self, filename, nb_images=None):
        """Load a dataset from a json file indicated by `filename` ; use dict
        comprehension instead of direct assignments in order to convert dict
        keys to integers

        Parameters
        ----------
        filename : str
            String designing the relative path from where the dataset must be
        loaded
        nb_images : integer
            Number of images that must be loaded (if None, the whole dataset is
        loaded)
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


class GeoreferencedDataset(Dataset):
    """Generic class that describes the behavior of "Dataset" objects
    corresponding to aerial images

    """

    def _generate_preprocessed_filenames(
        self, image_filename, output_dir, x, y, suffix=None
    ):
        """Generate preprocessed image and label filenames on the file system,
        starting from a raw image filename

        Parameters
        ----------
        image_filename : str
            Original image filename
        output_dir : str
            Output folder for preprocessed material
        x : int
            Extracted image west coordinates
        y : int
            Extracted image north coordinates
        suffix : str
            Preprocessed filename complement

        Returns
        -------
        dict
            Preprocessed image and corresponding label filenames
        """
        basename_decomp = os.path.splitext(os.path.basename(image_filename))
        img_id_str = (
            str(self.image_size)
            + "_"
            + str(self.image_size)
            + "_"
            + str(x)
            + "_"
            + str(y)
        )
        img_id_str = (
            img_id_str if suffix is None else img_id_str + "_" + suffix
        )
        new_filename = basename_decomp[0] + "_" + img_id_str + ".png"
        out_image_name = os.path.join(output_dir, "images", new_filename)
        out_label_name = out_image_name.replace("images", "labels")
        return {"image": out_image_name, "labels": out_label_name}

    def _serialize(
        self,
        tile_image,
        labelled_image,
        label_dict,
        image_filename,
        output_dir,
        x,
        y,
        suffix=None,
    ):
        """Serialize a tiled image generated from an original high-resolution
        raster as well as the labelled version of the tile

        The method returns a dict that contains image-related file paths.

        Parameters
        ----------
        tile_image : PIL.Image
        labelled_image : PIL.Image
        label_dict : dict
        image_filename : str
        output_dir : str
        x : int
        y : int

        Returns
        -------
        dict
            Information related to the serialized tile (file paths, encountered
        labels)
        """
        dirs = self._generate_preprocessed_filenames(
            image_filename, output_dir, x, y, suffix
        )
        try:
            tile_image.verify()
            labelled_image.verify()
            tile_image.save(dirs["image"])
            labelled_image.save(dirs["labels"])
            return {
                "raw_filename": image_filename,
                "image_filename": dirs["image"],
                "label_filename": dirs["labels"],
                "labels": label_dict,
            }
        except SyntaxError as se:
            logger.error(
                "The image %s is corrupt, hence not serialized.",
                image_filename
            )
            return None

    def _preprocess_tile(
        self, x, y, image_filename, output_dir, raster, labels=None
    ):
        """Preprocess one single tile built from `image_filename`, with respect
                         to pixel coordinates `(x, y)`

        Parameters
        ----------
        x : int
            Horizontal pixel coordinate (*i.e.* west bound)
        y : int
            Vertical pixel coordinate (*i.e.* north bound)
        image_filename : str
            Full path towards the image on the disk
        output_dir : str
            Output path where preprocessed image must be saved
        raster : osgeo.gdal.Dataset
            Original georeferenced raster
        labels : geopandas.GeoDataFrame
            Raw image labels (*i.e.* georeferenced buildings)

        Returns
        -------
        dict
            Key/values with the filenames and label ids

        """
        dirs = self._generate_preprocessed_filenames(
            image_filename, output_dir, x, y
        )
        gdal.Translate(
            dirs["image"],
            raster,
            format="PNG",
            srcWin=[x, y, self.image_size, self.image_size],
        )
        return {
            "raw_filename": image_filename,
            "image_filename": dirs["image"],
        }

    def _preprocess_for_inference(self, image_filename, output_dir):
        """Resize/crop then save the training & label images

        Parameters
        ----------
        image_filename : str
            Full path towards the image on the disk
        output_dir : str
            Output path where preprocessed image must be saved

        Returns
        -------
        dict
            Key/values with the filenames and label ids
        """
        raster = gdal.Open(image_filename)
        raw_img_width = raster.RasterXSize
        raw_img_height = raster.RasterYSize
        result_dicts = []
        logger.info("Image filename: %s", image_filename)
        logger.info("Raw image size: %s, %s", raw_img_width, raw_img_height)

        for x in range(0, raw_img_width, self.image_size):
            for y in range(0, raw_img_height, self.image_size):
                tile_results = self._preprocess_tile(
                    x, y, image_filename, output_dir, raster
                )
                result_dicts.append(tile_results)
        del raster
        return result_dicts


    def load_mask(self, buildings, raster_features, min_x, min_y):
        """Translate georeferenced buildings as numpy arrays in order to
        prepare image analysis

        Parameters
        ----------
        buildings : geopandas.GeoDataFrame
            Georeferenced building labels, with a ̀condition` column that
        contains building type and a `geometry` column that describe the
        geolocalization
        raster_features : dict
            Geographical features of raw original image
        min_x : int
            Minimal tile x-coordinates (west bound)
        min_y : int
            Minimal tile y-coordinates (north bound)

        Returns
        -------
        numpy.array
            Array-versionned building labels, that link each pixel to the
        specified class, *i.e.* `B(x, y)=i` if pixel `(x, y)` belongs to class
        `i`. In this dataset, the labels are `complete`, `incomplete` or
        ̀foundation`.

        """
        mask = np.zeros(
            shape=(self.image_size, self.image_size), dtype=np.uint8
        )
        if buildings.shape[0] == 0:
            return mask
        for idx, row in buildings.iterrows():
            points = geometries.extract_points_from_polygon(
                row["geometry"], raster_features, min_x, min_y
            )
            label_id = [
                label["id"]
                for label in self.labels
                if label["name"] == row["condition"].lower()
            ][0]
            mask = cv2.fillPoly(mask, [points], label_id)
        return mask


    def populate(
        self,
        output_dir,
        input_dir,
        nb_images=0,
        nb_tiles_per_image=0,
        labelling=True,
        nb_processes=1,
    ):
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
        nb_tiles_per_image : integer
            Number of tiles that must be picked into the raw image, for labelled datasets
        labelling : boolean
            If True labels are recovered from dataset, otherwise dummy label are generated
        nb_processes : int
            Number of processes on which to run the preprocessing
        """
        image_list = os.listdir(os.path.join(input_dir, "images"))
        image_list_longname = [
            os.path.join(input_dir, "images", l)
            for l in image_list
            if not l.startswith(".")
        ]
        nb_image_files = len(image_list_longname)
        if nb_image_files < nb_images:
            logger.warning(
                "Asking to preprocess %s images, but only got %s files",
                nb_images, nb_image_files)
            nb_images = nb_image_files
            logger.warning("Preprocessing %s images..", nb_images)
        image_list_longname = np.random.choice(
            image_list_longname, nb_images, replace=False
        )

        logger.info("Getting %s images to preprocess...", nb_images)
        if labelling:
            if nb_processes == 1:
                for x in image_list_longname:
                    self.image_info.append(
                        self._preprocess_for_training(
                            x, output_dir, nb_tiles_per_image
                        )
                    )
            else:
                with Pool(processes=nb_processes) as p:
                    self.image_info = p.starmap(
                        self._preprocess_for_training,
                        [
                            (x, output_dir, nb_tiles_per_image)
                            for x in image_list_longname
                        ],
                    )
        else:
            if nb_processes == 1:
                for x in image_list_longname:
                    self.image_info.append(
                        self._preprocess_for_inference(x, output_dir)
                    )
            else:
                with Pool(processes=nb_processes) as p:
                    self.image_info = p.starmap(
                        self._preprocess_for_inference,
                        [(x, output_dir) for x in image_list_longname],
                    )

        self.image_info = [
            item for sublist in self.image_info for item in sublist
        ]
        logger.info(
            "Saved %s images in the preprocessed dataset.",
            len(self.image_info),
        )

