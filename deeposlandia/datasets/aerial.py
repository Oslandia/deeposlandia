"""Aerial dataset module

Model Aerial images for convolutional neural network applications. Data is
downloadable at https://project.inria.fr/aerialimagelabeling/files/.

"""

from multiprocessing import Pool
import os

import daiquiri
import numpy as np
from osgeo import gdal
from PIL import Image

from deeposlandia.datasets import GeoreferencedDataset
from deeposlandia import geometries, utils


logger = daiquiri.getLogger(__name__)


class AerialDataset(GeoreferencedDataset):
    """Dataset structure inspired from AerialImageDataset, a dataset released
    by Inria

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
        # self.tile_size = tile_size
        # img_size = utils.get_image_size_from_tile(self.tile_size)
        super().__init__(tile_size)
        self.add_label(
            label_id=0, label_name="background", color=0, is_evaluate=True
        )
        self.add_label(
            label_id=1, label_name="building", color=255, is_evaluate=True
        )


    def _preprocess_for_training(self, image_filename, output_dir, nb_images):
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
        image_data = raster.ReadAsArray()
        image_data = np.swapaxes(image_data, 0, 2)
        result_dicts = []
        logger.info(
            "Image filename: %s, size: (%s, %s)",
            image_filename.split("/")[-1], raw_img_width, raw_img_height
        )

        label_filename = image_filename.replace("images", "labels")
        label_raster = gdal.Open(label_filename)
        labels = label_raster.ReadAsArray()
        labels = np.swapaxes(labels, 0, 1)

        nb_attempts = 0
        image_counter = 0
        empty_image_counter = 0
        while image_counter < nb_images and nb_attempts < 2 * nb_images:
            # randomly pick an image
            x = np.random.randint(0, raw_img_width - self.image_size)
            y = np.random.randint(0, raw_img_height - self.image_size)

            tile_data = image_data[
                x:(x + self.image_size), y:(y + self.image_size)
            ]
            tile_image = Image.fromarray(tile_data)
            mask = labels[
                x:(x + self.image_size), y:(y + self.image_size)
            ]
            label_dict = utils.build_labels(
                mask, range(self.get_nb_labels()), "aerial"
            )
            labelled_image = Image.fromarray(mask)
            if np.unique(mask).shape[0] > 1:
                tiled_results = self._serialize(
                    tile_image,
                    labelled_image,
                    label_dict,
                    image_filename,
                    output_dir,
                    x,
                    y,
                    "nw",
                )
                if tiled_results:
                    result_dicts.append(tiled_results)
                image_counter += 1
                tile_image_ne = tile_image.transpose(Image.FLIP_LEFT_RIGHT)
                labelled_image_ne = labelled_image.transpose(
                    Image.FLIP_LEFT_RIGHT
                )
                tiled_results_ne = self._serialize(
                    tile_image_ne,
                    labelled_image_ne,
                    label_dict,
                    image_filename,
                    output_dir,
                    x,
                    y,
                    "ne",
                )
                if tiled_results_ne:
                    result_dicts.append(tiled_results_ne)
                image_counter += 1
                tile_image_sw = tile_image.transpose(Image.FLIP_TOP_BOTTOM)
                labelled_image_sw = labelled_image.transpose(
                    Image.FLIP_TOP_BOTTOM
                )
                tiled_results_sw = self._serialize(
                    tile_image_sw,
                    labelled_image_sw,
                    label_dict,
                    image_filename,
                    output_dir,
                    x,
                    y,
                    "sw",
                )
                if tiled_results_sw:
                    result_dicts.append(tiled_results_sw)
                image_counter += 1
                tile_image_se = tile_image_sw.transpose(Image.FLIP_LEFT_RIGHT)
                labelled_image_se = labelled_image_sw.transpose(
                    Image.FLIP_LEFT_RIGHT
                )
                tiled_results_se = self._serialize(
                    tile_image_se,
                    labelled_image_se,
                    label_dict,
                    image_filename,
                    output_dir,
                    x,
                    y,
                    "se",
                )
                if tiled_results_se:
                    result_dicts.append(tiled_results_se)
                image_counter += 1
                del tile_image_se, tile_image_sw, tile_image_ne
                del labelled_image_se, labelled_image_sw, labelled_image_ne
            else:
                if empty_image_counter < 0.1 * nb_images:
                    tiled_results = self._serialize(
                        tile_image,
                        labelled_image,
                        label_dict,
                        image_filename,
                        output_dir,
                        x,
                        y,
                        "nw",
                    )
                    if tiled_results:
                        result_dicts.append(tiled_results)
                    image_counter += 1
                    empty_image_counter += 1
            nb_attempts += 1
        del raster
        logger.info(
            "Generate %s images after %s attempts.", image_counter, nb_attempts
        )
        return result_dicts
