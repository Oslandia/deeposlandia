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
        "-a",
        "--aggregate-label",
        action="store_true",
        help="Aggregate labels with respect to their categories",
    )
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
        default=224,
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

        Build a high-resolution image labelling by
        predicting semantic segmentation labels on
        image patches, and by postprocessing resulting
        arrays so as to get geographic entities.

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
        type=int,
        help=("Number of images in each inference batch"),
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
    """
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
    """
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


def add_training_args(parser):
    """
    """
    parser.add_argument(
        "-a",
        "--aggregate-label",
        action="store_true",
        help="Aggregate labels with respect to their categories",
    )
    parser.add_argument(
        "-M",
        "--model",
        default="feature_detection",
        help=(
            "Type of model to train, either "
            "'feature_detection' or 'semantic_segmentation'"
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
        default=["simple"],
        nargs="+",
        help=(
            "Neural network size, either 'simple', 'vgg', "
            "'inception' or 'unet' ('simple' refers to 3 "
            "conv/pool blocks and 1 fully-connected layer; "
            "the others refer to state-of-the-art networks)"
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
