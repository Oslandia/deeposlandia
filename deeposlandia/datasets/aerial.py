"""Aerial dataset module

Model Aerial images for convolutional neural network applications. Data is
downloadable at https://project.inria.fr/aerialimagelabeling/files/.

"""

from multiprocessing import Pool
import os

import daiquiri
import numpy as np
from PIL import Image

from deeposlandia.datasets import Dataset
from deeposlandia import utils


logger = daiquiri.getLogger(__name__)


class AerialDataset(Dataset):
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
        self.tile_size = tile_size
        img_size = utils.get_image_size_from_tile(self.tile_size)
        super().__init__(img_size)
        self.add_label(
            label_id=0, label_name="background", color=0, is_evaluate=True
        )
        self.add_label(
            label_id=1, label_name="building", color=255, is_evaluate=True
        )

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
                tile = img_in.crop(
                    (x, y, x + self.tile_size, y + self.tile_size)
                )
                tile = utils.resize_image(tile, self.image_size)
                img_id = int(
                    (
                        raw_img_size / self.tile_size * x / self.tile_size
                        + y / self.tile_size
                    )
                )
                basename_decomp = os.path.splitext(
                    os.path.basename(image_filename)
                )
                new_in_filename = (
                    basename_decomp[0] + "_" + str(img_id) + basename_decomp[1]
                )
                new_in_path = os.path.join(
                    output_dir, "images", new_in_filename
                )
                tile.save(new_in_path.replace(".tif", ".png"))
                result_dicts.append(
                    {
                        "raw_filename": image_filename,
                        "image_filename": new_in_path,
                    }
                )

        if labelling:
            label_filename = image_filename.replace("images/", "gt/")
            img_out = Image.open(label_filename)
            buffer_tiles = []
            for x in range(0, raw_img_size, self.tile_size):
                for y in range(0, raw_img_size, self.tile_size):
                    tile = img_out.crop(
                        (x, y, x + self.tile_size, y + self.tile_size)
                    )
                    tile = utils.resize_image(tile, self.image_size)
                    img_id = int(
                        (
                            raw_img_size / self.tile_size * x / self.tile_size
                            + y / self.tile_size
                        )
                    )
                    basename_decomp = os.path.splitext(
                        os.path.basename(image_filename)
                    )
                    new_out_filename = (
                        basename_decomp[0]
                        + "_"
                        + str(img_id)
                        + basename_decomp[1]
                    )
                    new_out_path = os.path.join(
                        output_dir, "labels", new_out_filename
                    )
                    tile.save(new_out_path.replace(".tif", ".png"))
                    labels = utils.build_labels(
                        tile, self.label_ids, dataset="aerial"
                    )
                    result_dicts[img_id]["label_filename"] = new_out_path
                    result_dicts[img_id]["labels"] = labels

        return result_dicts

    def populate(
        self,
        output_dir,
        input_dir,
        nb_images=None,
        aggregate=False,
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
        aggregate : bool
            Label aggregation parameter, useless for this dataset, but kept for
        class method genericity
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
        ][:nb_images]
        logger.info(
            "Getting %s images to preprocess...", len(image_list_longname)
        )
        if nb_processes == 1:
            for x in image_list_longname:
                self.image_info.append(
                    self._preprocess(x, output_dir, labelling)
                )
        else:
            with Pool(processes=nb_processes) as p:
                self.image_info = p.starmap(
                    self._preprocess,
                    [(x, output_dir, labelling) for x in image_list_longname],
                )
        self.image_info = [
            item for sublist in self.image_info for item in sublist
        ]
        logger.info(
            "Saved %s images in the preprocessed dataset.",
            len(self.image_info),
        )
