"""Mapillary dataset module

Model Mapillary street-scene images for convolutional neural network
applications. Data is downloadable at https://www.mapillary.com/dataset/vistas.

"""

from multiprocessing import Pool
import os

import daiquiri
import numpy as np
from PIL import Image

from deeposlandia.datasets import Dataset
from deeposlandia import utils


logger = daiquiri.getLogger(__name__)


class MapillaryDataset(Dataset):
    """Dataset structure that gathers all information related to the Mapillary
    images

    Attributes
    ----------
    image_size : int
        Size of considered images (height=width), raw images will be resized
    during the preprocessing
    glossary_filename : str
        Name of the Mapillary input glossary, that contains every information
    about Mapillary labels

    """

    def __init__(self, image_size, glossary_filename):
        """ Class constructor ; instanciates a MapillaryDataset as a standard
        Dataset which is completed by a glossary file that describes the
        dataset labels
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
        glossary = utils.read_config(config_filename)
        if "labels" not in glossary:
            logger.error("There is no 'label' key in the provided glossary.")
            return None
        for lab_id, label in enumerate(glossary["labels"]):
            self.add_label(
                lab_id,
                label["name"],
                label["color"],
                label["evaluate"],
                label["family"],
                label["contains_id"],
                label["contains"]
            )

    def group_image_label(self, image):
        """Group the labels

        If the label ids 4, 5 and 6 belong to the same group, they will be
        turned into the label id 4.

        Parameters
        ----------
        image : PIL.Image

        Returns
        -------
        PIL.Image
        """
        # turn all label ids into the lowest digits/label id
        # according to its "group" (manually built)
        a = np.array(image)
        for root_id, label in enumerate(self.label_info):
            for label_id in label.get("aggregate"):
                mask = a == label_id
                a[mask] = root_id
        return Image.fromarray(a, mode=image.mode)

    def _preprocess(
        self, image_filename, output_dir, labelling=True
    ):
        """Resize/crop then save the training & label images

        Parameters
        ----------
        datadir : str
        image_filaname : str
        labelling : boolean

        Returns
        -------
        dict
            Key/values with the filenames and label ids
        """
        # open original images
        img_in = Image.open(image_filename)

        # resize images
        # (self.image_size*larger_size or larger_size*self.image_size)
        img_in = utils.resize_image(img_in, self.image_size)

        # crop images to get self.image_size*self.image_size dimensions
        crop_pix = np.random.randint(0, 1 + max(img_in.size) - self.image_size)
        final_img_in = utils.mono_crop_image(img_in, crop_pix)

        # save final image
        new_in_filename = os.path.join(
            output_dir, "images", os.path.basename(image_filename)
        )
        final_img_in.save(new_in_filename)

        # label_filename vs label image
        if labelling:
            label_filename = image_filename.replace("images/", "labels/")
            label_filename = label_filename.replace(".jpg", ".png")
            img_out = Image.open(label_filename)
            img_out = utils.resize_image(img_out, self.image_size)
            img_out = utils.mono_crop_image(img_out, crop_pix)
            # aggregate some labels
            img_out = self.group_image_label(img_out)

            labels = utils.build_labels(
                img_out, self.label_ids, dataset="mapillary"
            )
            new_out_filename = os.path.join(
                output_dir, "labels", os.path.basename(label_filename)
            )
            label_out = np.array(img_out)
            final_img_out = utils.build_image_from_config(
                label_out, self.label_info
            )
            final_img_out.save(new_out_filename)
        else:
            new_out_filename = None
            labels = {i: 0 for i in range(self.get_nb_labels())}

        return {
            "raw_filename": image_filename,
            "image_filename": new_in_filename,
            "label_filename": new_out_filename,
            "labels": labels,
        }

    def populate(
        self,
        output_dir,
        input_dir,
        nb_images=None,
        nb_tiles_per_image=None,
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
            Number of tiles that must be picked into the raw image (useless there, added
        for consistency)
        labelling: boolean
            If True labels are recovered from dataset, otherwise dummy label are generated
        nb_processes : int
            Number of processes on which to run the preprocessing
        """
        if nb_tiles_per_image is not None:
            logger.warning("The ``nb_tiles_per_image`` parameter is useless, it will be ignored.")
        image_list = os.listdir(os.path.join(input_dir, "images"))[:nb_images]
        image_list_longname = [
            os.path.join(input_dir, "images", l) for l in image_list
        ]
        if nb_processes == 1:
            for x in image_list_longname:
                self.image_info.append(
                    self._preprocess(x, output_dir, labelling)
                )
        else:
            with Pool(processes=nb_processes) as p:
                self.image_info = p.starmap(
                    self._preprocess,
                    [
                        (x, output_dir, labelling)
                        for x in image_list_longname
                    ],
                )
