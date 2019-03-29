"""Tanzania building dataset class

To add building instances into the dataset, use the load_buildings() method by
passing as the first parameter the folder that contains building informations
(images, features, items)

"""

from multiprocessing import Pool
import os

import cv2
import daiquiri
import geopandas as gpd
import numpy as np
from osgeo import gdal
from PIL import Image

from deeposlandia.datasets import GeoreferencedDataset
from deeposlandia import geometries, utils


logger = daiquiri.getLogger(__name__)

# Save png tiles without auxiliary information on disk
os.environ["GDAL_PAM_ENABLED"] = "NO"


class TanzaniaDataset(GeoreferencedDataset):
    """Tanzania building dataset, as released during the Open AI Tanzania
    challenge

    See:
    https://blog.werobotics.org/2018/08/06/welcome-to-the-open-ai-tanzania-challenge/
    The dataset is composed of 20 high-resolution images (~6-8cm/pixel, until
    76k*76k pixels), 13 being associated with geo-referenced labels, 7 for
    testing purpose.

    Attributes
    ----------
    img_size : int
        Size of the tiles into which each raw images is decomposed during
    dataset population (height=width)

    """

    BACKGROUND_COLOR = [0, 0, 0]
    COMPLETE_COLOR = [50, 200, 50]
    INCOMPLETE_COLOR = [200, 200, 50]
    FOUNDATION_COLOR = [200, 50, 50]

    def __init__(self, img_size):
        """Class constructor ; instanciates a TanzaniaDataset as a standard Dataset
        which is completed by a glossary file that describes the dataset labels
        and images

        """
        super().__init__(img_size)
        self.add_label(
            label_id=0,
            label_name="background",
            color=self.BACKGROUND_COLOR,
            is_evaluate=True,
        )
        self.add_label(
            label_id=1,
            label_name="complete",
            color=self.COMPLETE_COLOR,
            is_evaluate=True,
        )
        self.add_label(
            label_id=2,
            label_name="incomplete",
            color=self.INCOMPLETE_COLOR,
            is_evaluate=True,
        )
        self.add_label(
            label_id=3,
            label_name="foundation",
            color=self.FOUNDATION_COLOR,
            is_evaluate=True,
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

        label_filename = image_filename.replace("images", "labels").replace(
            ".tif", ".geojson"
        )
        labels = gpd.read_file(label_filename)
        labels = labels.loc[~labels.geometry.isna(), ["condition", "geometry"]]
        none_mask = [lc is None for lc in labels.condition]
        labels.loc[none_mask, "condition"] = "Complete"

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
            raster_features = geometries.get_image_features(raster)
            tile_items = geometries.extract_tile_items(
                raster_features, labels, x, y, self.image_size, self.image_size
            )
            mask = self.load_mask(tile_items, raster_features, x, y)
            label_dict = utils.build_labels(
                mask, range(self.get_nb_labels()), "tanzania"
            )
            labelled_image = utils.build_image_from_config(mask, self.labels)
            if len(tile_items) > 0:
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
