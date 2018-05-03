
"""Inference on deep learning model previously trained

* Build the instance name

* Load one (or more) image(s) from the file system:

* Load a trained model starting from the instance name

* Make label predictions on the test image(s)

* Produce a result: for instance, only predicted labels

Example of program call, that will infers labels on all files from
̀path_to_images/shapes_00000.png` to `path_to_images/shapes_00009.png`::

    python deeposlandia/inference.py -D shapes -i path_to_images/shapes_0000*.png

"""

import argparse
import glob
import numpy as np
import os
from PIL import Image
import sys

from keras.models import Model

from deeposlandia import utils
from deeposlandia.feature_detection import FeatureDetectionNetwork
from deeposlandia.semantic_segmentation import SemanticSegmentationNetwork


def add_program_arguments(parser):
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
    parser.add_argument('-D', '--dataset',
                        required=True,
                        help="Dataset type (either mapillary or shapes)")
    parser.add_argument('-i', '--image-paths',
                        required=True,
                        nargs='+',
                        help="Path of the image on the file system")
    parser.add_argument('-M', '--model',
                        default="feature_detection",
                        help=("Type of model to train, either "
                              "'feature_detection' or 'semantic_segmentation'"))
    parser.add_argument('-p', '--datapath',
                        default="./data",
                        help="Relative path towards data directory")
    return parser

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
    parser.add_argument('-b', '--batch-size',
                        type=int,
                        default=None,
                        help=("Number of images that must be contained "
                              "into a single batch"))
    parser.add_argument('-d', '--dropout',
                        type=float,
                        default=None,
                        help="Percentage of kept neurons during training")
    parser.add_argument('-L', '--learning-rate', 
                        default=None,
                        type=float,
                        help=("Starting learning rate"))
    parser.add_argument('-l', '--learning-rate-decay',
                        default=None,
                        type=float,
                        help=("Learning rate decay"))
    parser.add_argument('-n', '--name',
                        default=None,
                        help=("Model name that will be used for results, "
                              "checkout and graph storage on file system"))
    parser.add_argument('-N', '--network',
                        default=None,
                        help=("Neural network size, either 'simple', 'vgg' or "
                              "'inception' ('simple' refers to 3 conv/pool "
                              "blocks and 1 fully-connected layer; the others "
                              "refer to state-of-the-art networks)"))
    return parser


if __name__ == '__main__':

    program_description = ("Infer labels on one (or more) image file(s) "
                           "from a trained deep neural network")
    parser = argparse.ArgumentParser(description=program_description)
    parser = add_program_arguments(parser)
    parser = add_instance_arguments(parser)
    args = parser.parse_args()

    # `image_paths` is first got as [[image1, ..., image_i], [image_j, ..., image_n]]
    image_paths = [glob.glob(f) for f in args.image_paths]
    # then it is flattened to get a simple list
    flattened_image_paths = sum(image_paths, [])
    x_test = []
    for image_path in flattened_image_paths:
        image = Image.open(image_path)
        image_size = image.size[0]
        if image.size[0] != image.size[1]:
            utils.logger.error(("One of the parsed images "
                                "has non-squared dimensions."))
            sys.exit(1)
        x_test.append(np.array(image))
    x_test = np.array(x_test)

    aggregate_value = "full" if not args.aggregate_label else "aggregated"
    instance_args = [args.name, image_size, args.network, args.batch_size,
                     aggregate_value, args.dropout,
                     args.learning_rate, args.learning_rate_decay]
    instance_name = utils.list_to_str(instance_args, "_")

    input_folder = utils.prepare_input_folder(args.datapath, args.dataset)
    prepro_folder = utils.prepare_preprocessed_folder(args.datapath,
                                                      args.dataset,
                                                      image_size,
                                                      aggregate_value)

    if os.path.isfile(prepro_folder["training_config"]):
        train_config = utils.read_config(prepro_folder["training_config"])
        label_ids = [x['id'] for x in train_config['labels']
                     if x['is_evaluate']]
        nb_labels = len(label_ids)
    else:
        utils.logger.error(("There is no valid data with the specified parameters. "
                           "Please generate a valid dataset "
                            "before calling the program."))
        sys.exit(1)

    # Model creation
    if args.model == "feature_detection":
        net = FeatureDetectionNetwork(network_name=instance_name,
                                      image_size=image_size,
                                      nb_labels=nb_labels,
                                      dropout=args.dropout,
                                      architecture=args.network)
        loss_function = "binary_crossentropy"
    elif args.model == "semantic_segmentation":
        net = SemanticSegmentationNetwork(network_name=instance_name,
                                          image_size=image_size,
                                          nb_labels=nb_labels,
                                          dropout=args.dropout,
                                          architecture=args.network)
        loss_function = "categorical_crossentropy"
    else:
        utils.logger.error(("Unrecognized model. Please enter 'feature_detection' "
                            "or 'semantic_segmentation'."))
        sys.exit(1)
    model = Model(net.X, net.Y)

    if any([arg is None for arg in instance_args]):
        utils.logger.info("Some arguments are None, the best model is considered.")
        output_folder = utils.prepare_output_folder(args.datapath,
                                                    args.dataset,
                                                    args.model)
        checkpoints = [item for item in os.listdir(output_folder)
                       if os.path.isfile(os.path.join(output_folder, item))]
        if len(checkpoints) > 0:
            model_checkpoint = max(checkpoints)
            checkpoint_complete_path = os.path.join(output_folder,
                                                    model_checkpoint)
            model.load_weights(checkpoint_complete_path)
            utils.logger.info(("Model weights have been recovered from {}"
                               "").format(checkpoint_complete_path))
        else:
            utils.logger.info(("No available trained model for this image size"
                               " with optimized hyperparameters. The "
                               "inference will be done on an untrained model"))
    else:
        utils.logger.info("All instance arguments are filled out.")
        output_folder = utils.prepare_output_folder(args.datapath,
                                                    args.dataset,
                                                    args.model,
                                                    instance_name)
        model_checkpoint = "best-model-" + str(image_size) + ".h5"
        checkpoint_complete_path = os.path.join(output_folder,
                                                model_checkpoint)
        if os.path.isfile(checkpoint_complete_path):
            model.load_weights(checkpoint_complete_path)
            utils.logger.info(("Model weights have been recovered from {}"
                               "").format(checkpoint_complete_path))
        else:
            utils.logger.info(("No available checkpoint for this configuration. "
                               "The model will be trained from scratch."))

    y_raw_pred = model.predict(x_test)
    y_pred = np.round(y_raw_pred).astype(np.uint8)
    utils.logger.info("Predicted labels:")
    for image, prediction in zip(flattened_image_paths, y_pred.tolist()):
        utils.logger.info("{}: {}".format(image, prediction))
