"""Main method to generate new datasets

Example of program calls:

* generate 64*64 pixel images from Shapes dataset, 10000 images in the training set, 100 in the
validation set, 1000 in the testing set::

    python deeposlandia/datagen.py -D shapes -s 64 -t 10000 -v 100 -T 1000

* generate 224*224 pixel images from Mapillary dataset, with aggregated labels::

    python deeposlandia/datagen.py -D mapillary -s 224 -a

"""

import argparse
import os
import sys

import pandas as pd

from deeposlandia import utils
from deeposlandia.dataset import MapillaryDataset, ShapeDataset

def add_instance_arguments(parser):
    """Add instance-specific arguments from the command line

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Input parser

    Returns
    -------
    argparse.ArgumentParser
        Modified parser, with additional arguments
    """
    parser.add_argument('-a', '--aggregate-label', action='store_true',
                        help="Aggregate labels with respect to their categories")
    parser.add_argument('-D', '--dataset',
                        required=True,
                        help="Dataset type (either mapillary or shapes)")
    parser.add_argument('-p', '--datapath',
                        default="./data",
                        help="Relative path towards data directory")
    parser.add_argument('-s', '--image-size',
                        default=256,
                        type=int,
                        help=("Desired size of images (width = height)"))
    parser.add_argument('-T', '--nb-testing-image',
                        type=int,
                        default=5000,
                        help=("Number of testing images"))
    parser.add_argument('-t', '--nb-training-image',
                        type=int,
                        default=18000,
                        help=("Number of training images"))
    parser.add_argument('-v', '--nb-validation-image',
                        type=int,
                        default=200,
                        help=("Number of validation images"))
    return parser


if __name__=='__main__':

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description=("Convolutional Neural Netw"
                                                  "ork on street-scene images"))
    parser = add_instance_arguments(parser)
    args = parser.parse_args()

    # Data path and repository management
    aggregate_value = "full" if not args.aggregate_label else "aggregated"
    folders = utils.prepare_folders(args.datapath, args.dataset, aggregate_value,
                                    args.image_size, "")

    # Dataset creation
    if args.dataset == "mapillary":
        config_name = "config.json" if not args.aggregate_label else "config_aggregate.json"
        config_path = os.path.join(folders["input"], config_name)
        train_dataset = MapillaryDataset(args.image_size, config_path)
        validation_dataset = MapillaryDataset(args.image_size, config_path)
        test_dataset = MapillaryDataset(args.image_size, config_path)
    elif args.dataset == "shapes":
        train_dataset = ShapeDataset(args.image_size)
        validation_dataset = ShapeDataset(args.image_size)
        test_dataset = ShapeDataset(args.image_size)
    else:
        utils.logger.error("Unsupported dataset type. Please choose 'mapillary' or 'shapes'")
        sys.exit(1)

    # Dataset populating/loading (depends on the existence of a specification file)
    if os.path.isfile(folders["training_config"]):
        train_dataset.load(folders["training_config"], args.nb_training_image)
    else:
        utils.logger.info(("No existing configuration file for this dataset. Create {}"
                           "").format(folders["training_config"]))
        input_image_dir = os.path.join(folders["input"], "training")
        train_dataset.populate(folders["prepro_training"], input_image_dir,
                               nb_images=args.nb_training_image,
                               aggregate=args.aggregate_label)
        train_dataset.save(folders["training_config"])
    if os.path.isfile(folders["validation_config"]):
        validation_dataset.load(folders["validation_config"], args.nb_validation_image)
    else:
        utils.logger.info(("No existing configuration file for this dataset. Create {}"
                           "").format(folders["validation_config"]))
        input_image_dir = os.path.join(folders["input"], "validation")
        validation_dataset.populate(folders["prepro_validation"], input_image_dir,
                                    nb_images=args.nb_validation_image,
                                    aggregate=args.aggregate_label)
        validation_dataset.save(folders["validation_config"])
    if os.path.isfile(folders["testing_config"]):
        test_dataset.load(folders["testing_config"], args.nb_testing_image)
    else:
        utils.logger.info(("No existing configuration file for this dataset. Create {}"
                           "").format(folders["testing_config"]))
        input_image_dir = os.path.join(folders["input"], "testing")
        test_dataset.populate(folders["prepro_testing"], input_image_dir,
                                    nb_images=args.nb_testing_image,
                                    aggregate=args.aggregate_label)
        test_dataset.save(folders["testing_config"])

    glossary = pd.DataFrame(train_dataset.labels)
    glossary["popularity"] = train_dataset.get_label_popularity()
    utils.logger.info("Data glossary:\n{}".format(glossary))
    sys.exit(0)
