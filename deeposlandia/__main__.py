"""Deeposlandia central command-line interface

Available choices::
    - deepo datagen [args]
    - deepo train [args]
    - deepo infer [args]
    - deepo postprocess [args]
"""

import argparse
import os

from deeposlandia import datagen, train, inference, postprocess
from deeposlandia import AVAILABLE_MODELS
from deeposlandia.datasets import AVAILABLE_DATASETS, GEOGRAPHIC_DATASETS


def datagen_parser(subparser, reference_func):
    """Add arguments focused on data generation process

    Parameters
    ----------
    subparser : argparser.parser.SubParsersAction
    reference_func : function
    """
    parser = subparser.add_parser(
        "datagen",
        help="Preprocess new datasets for automated image analysis"
    )
    add_dataset_args(parser)
    add_nb_image_args(parser)
    parser.add_argument(
        "-s",
        "--image-size",
        default=256,
        type=int,
        help=("Desired size of images (width = height)"),
    )
    parser.add_argument(
        "-T",
        "--nb-testing-image",
        type=int,
        default=0,
        help=("Number of testing images"),
    )
    parser.set_defaults(func=reference_func)


def train_parser(subparser, reference_func):
    """Add instance-specific arguments from the command line

    Parameters
    ----------
    subparser : argparser.parser.SubParsersAction
    reference_func : function
    """
    parser = subparser.add_parser(
        "train",
        help="Train convolutional neural networks for automated image analysis"
    )
    add_dataset_args(parser)
    add_nb_image_args(parser)
    add_training_args(parser)
    parser.add_argument(
        "-s",
        "--image-size",
        default=256,
        type=int,
        help=("Desired size of images (width = height)"),
    )
    parser.add_argument(
        "-e",
        "--nb-epochs",
        type=int,
        default=0,
        help=(
            "Number of training epochs (one epoch means "
            "scanning each training image once)"
        ),
    )
    parser.set_defaults(func=reference_func)


def inference_parser(subparser, reference_func):
    """Add instance-specific arguments from the command line

    Parameters
    ----------
    subparser : argparser.parser.SubParsersAction
    reference_func : function
    """
    parser = subparser.add_parser(
        "infer",
        help="Predict image labelling with a neural network model"
    )
    add_dataset_args(parser)
    add_training_args(parser)
    parser.add_argument(
        "-i",
        "--image-paths",
        required=True,
        nargs="+",
        help="Path of the image on the file system",
    )
    parser.set_defaults(func=reference_func)


def postprocess_parser(subparser, reference_func):
    """Add instance-specific arguments from the command line

    Build a high-resolution image labelling by predicting semantic segmentation
    labels on image patches, and by postprocessing resulting arrays so as to
    get geographic entities.

    Parameters
    ----------
    subparser : argparser.parser.SubParsersAction
    reference_func : function
    """
    parser = subparser.add_parser(
        "postprocess",
        help=(
            "Postprocess the neural network prediction "
            + "(for geographic datasets only)"
        )
    )
    add_dataset_args(parser, available_datasets=GEOGRAPHIC_DATASETS)
    parser.add_argument(
        "-b",
        "--batch-size",
        default=2,
        type=int,
        help=("Number of images in each inference batch"),
    )
    parser.add_argument(
        "-g",
        "--draw-grid",
        action="store_true",
        help="If specified, draw the grid that materializes the predicted tiles"
    )
    parser.add_argument(
        "-i",
        "--image-basename",
        required=True,
        help="Basename of the image within the dataset",
    )
    parser.add_argument(
        "-s",
        "--image-size",
        required=True,
        type=int,
        help="Image patch size, in pixels",
    )
    parser.set_defaults(func=reference_func)


def add_dataset_args(parser, available_datasets=AVAILABLE_DATASETS):
    """Add arguments to "parser" that are related to the dataset material
    identification on the file system

    There are:
      - "datapath" that indicates where the datasets are stored on the file
    system
      - "dataset" gives the name of the dataset (which is also the dataset
    folder name on the file system)

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser that must be "augmented"
    available_datasets : tuple
        List of available datasets for the related command
    """
    parser.add_argument(
        "-P",
        "--datapath",
        default="./data",
        help="Relative path towards data directory",
    )
    parser.add_argument(
        "-D",
        "--dataset",
        required=True,
        choices=available_datasets,
        help="Dataset type",
    )


def add_nb_image_args(parser):
    """Add arguments to "parser" that are related to the amount of preprocessed
    data to prepare

    There are:
      - "nb_training_image" indicates the number of training image to process
      - "nb_validation_image" indicates the number of images to scan at each
    validation step

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser that must be "augmented"
    available_datasets : tuple
        List of available datasets for the related command
    """
    parser.add_argument(
        "-t",
        "--nb-training-image",
        type=int,
        default=0,
        help=("Number of training images"),
    )
    parser.add_argument(
        "-v",
        "--nb-validation-image",
        type=int,
        default=0,
        help=("Number of validation images"),
    )
    parser.add_argument(
        "--nb-tiles-per-image",
        type=int,
        help=(
            "Number of tile per raw image (mandatory for geographic datasets "
            "when training and/or validation images are involved)."
        ),
    )


def add_training_args(parser):
    """Add arguments to "parser" that are related to the convolutional neural
    network model training

    Such arguments are defined as follows:
      - "model" gives the problem that is solved, either semantic segmentation
    or feature detection
      - "name" refers to the instance name, for identification on the file
    system
      - "network" is the first hyperparameter, that describes the neural
    network architecture
      - "batch_size" is the second hyperparameter, namely the number of image
    in each training/validation batch
      - "dropout" indicates the rates of neurons that must be shutted down
    during each iteration, in order to reduce overfitting (third hyperparameter)
      - "learning_rate" gives the learning rate of the training process (fourth
    hyperparameter)
      - "learning_rate_decay" gives the decay rate of the learning rate
    throughout the training process (fifth hyperparameter)

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser that must be "augmented"
    """
    parser.add_argument(
        "-M",
        "--model",
        choices=AVAILABLE_MODELS,
        required=True,
        help=(
            "Type of model to train, either "
            "'featdet' (feature detection) or 'semseg' (semantic segmentation)"
        ),
    )
    parser.add_argument(
        "-n",
        "--name",
        default="cnn",
        help=(
            "Model name that will be used for results, "
            "checkout and graph storage on file system"
        ),
    )
    parser.add_argument(
        "-N",
        "--network",
        choices=["simple", "vgg", "resnet", "inception", "unet", "dilated"],
        default=["simple"],
        nargs="+",
        help=(
            "Neural network architecture, either 'simple', 'vgg', "
            "'inception' or 'resnet' for feature detection, and 'either', "
            "'unet' or 'dilated' for semantic segmentation "
            "('simple' refers to 3 conv/pool blocks and 1 fully-connected "
            "layer; the others refer to state-of-the-art networks)"
        ),
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        nargs="+",
        default=[50],
        help=(
            "Number of images that must be contained " "into a single batch"
        ),
    )
    parser.add_argument(
        "-d",
        "--dropout",
        nargs="+",
        default=[1.0],
        type=float,
        help="Percentage of kept neurons during training",
    )
    parser.add_argument(
        "-L",
        "--learning-rate",
        nargs="+",
        default=[0.001],
        type=float,
        help=("Starting learning rate"),
    )
    parser.add_argument(
        "-l",
        "--learning-rate-decay",
        nargs="+",
        default=[0.0001],
        type=float,
        help=("Learning rate decay"),
    )


def main():
    """Main method of the module
    """
    parser = argparse.ArgumentParser(
        prog="deepo",
        description="Deeposlandia framework for automated image analysis",
    )
    sub_parsers = parser.add_subparsers(dest="command")
    datagen_parser(sub_parsers, reference_func=datagen.main)
    train_parser(sub_parsers, reference_func=train.main)
    inference_parser(sub_parsers, reference_func=inference.main)
    postprocess_parser(sub_parsers, reference_func=postprocess.main)
    args = parser.parse_args()

    if args.func:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
