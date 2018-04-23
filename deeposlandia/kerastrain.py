"""Main method to train neural networks with Keras API
"""

import argparse
import os
import sys
import numpy as np

from datetime import datetime

from keras import backend
from keras.models import Model

from deeposlandia import generator, utils
from deeposlandia.keras_feature_detection import FeatureDetectionNetwork
from deeposlandia.keras_semantic_segmentation import SemanticSegmentationNetwork

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
    parser.add_argument('-M', '--model',
                        default="feature_detection",
                        help=("Type of model to train, either "
                              "'feature_detection' or 'semantic_segmentation'"))
    parser.add_argument('-n', '--name',
                        default="cnn",
                        help=("Model name that will be used for results, "
                              "checkout and graph storage on file system"))
    parser.add_argument('-N', '--network',
                        default='simple',
                        help=("Neural network size, either 'simple', 'vgg' or "
                              "'inception' ('simple' refers to 3 conv/pool "
                              "blocks and 1 fully-connected layer; the others "
                              "refer to state-of-the-art networks)"))
    parser.add_argument('-p', '--datapath',
                        default="./data",
                        help="Relative path towards data directory")
    return parser

def add_hyperparameters(parser):
    """Add hyperparameter arguments from the command line

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Input parser

    Returns
    -------
    argparse.ArgumentParser
        Modified parser, with additional arguments
    """
    parser.add_argument('-b', '--batch-size',
                        type=int,
                        default=20,
                        help=("Number of images that must be contained "
                              "into a single batch"))
    parser.add_argument('-d', '--dropout',
                        type=float,
                        default=1.0,
                        help=("Percentage of dropped out neurons "
                              "during training"))
    parser.add_argument('-e', '--nb-epochs',
                        type=int,
                        default=0,
                        help=("Number of training epochs (one epoch means "
                              "scanning each training image once)"))
    parser.add_argument('-l', '--learning-rate',
                        nargs="+",
                        default=[0.001, 300, 0.95],
                        type=float,
                        help=("List of learning rate components (starting LR, "
                              "decay steps and decay rate)"))
    parser.add_argument('-s', '--image-size',
                        default=256,
                        type=int,
                        help=("Desired size of images (width = height)"))
    return parser

def add_training_arguments(parser):
    """Add training-specific arguments from the command line

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Input parser

    Returns
    -------
    argparse.ArgumentParser
        Modified parser, with additional arguments
    """
    parser.add_argument('-ii', '--nb-testing-image',
                        type=int,
                        default=5000,
                        help=("Number of training images"))
    parser.add_argument('-it', '--nb-training-image',
                        type=int,
                        default=18000,
                        help=("Number of training images"))
    parser.add_argument('-iv', '--nb-validation-image',
                        type=int,
                        default=2000,
                        help=("Number of validation images"))
    return parser

if __name__=='__main__':
    """Main method: run a convolutional neural network using Keras API
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description=("Convolutional Neural Netw"
                                                  "ork on street-scene images"))
    parser = add_instance_arguments(parser)
    parser = add_hyperparameters(parser)
    parser = add_training_arguments(parser)
    args = parser.parse_args()

    # Data path and repository management
    aggregate_value = "full" if not args.aggregate_label else "aggregated"
    folders = utils.prepare_folders(args.datapath, args.dataset, aggregate_value,
                                    args.image_size, args.model)

    # Instance name (name + image size + network size + batch_size
    # + aggregate? + dropout + learning_rate)
    instance_args = [args.name, args.image_size, args.network, args.batch_size,
                     aggregate_value, args.dropout, utils.list_to_str(args.learning_rate)]
    instance_name = utils.list_to_str(instance_args, "_")

    # Data gathering
    train_seed = int(datetime.now().timestamp())
    if (os.path.isfile(folders["training_config"]) and os.path.isfile(folders["validation_config"])
        and os.path.isfile(folders["testing_config"])):
        train_config = utils.read_config(folders["training_config"])
        label_ids = [x['id'] for x in train_config['labels'] if x['is_evaluate']]
        train_generator = generator.create_generator(
            args.dataset,
            args.model,
            folders["prepro_training"],
            args.image_size,
            args.batch_size,
            label_ids,
            seed=train_seed)
        validation_generator = generator.create_generator(
            args.dataset,
            args.model,
            folders["prepro_validation"],
            args.image_size,
            args.batch_size,
            label_ids,
            seed=train_seed)
        test_generator = generator.create_generator(
            args.dataset,
            args.model,
            folders["prepro_testing"],
            args.image_size,
            args.batch_size,
            label_ids,
            inference=True,
            seed=train_seed)
    else:
        utils.logger.error(("There is no valid data with the specified parameters. "
                           "Please generate a valid dataset before calling the training program."))
        sys.exit(1)
    nb_labels = len(label_ids)

    # Model creation
    if args.model == "feature_detection":
        net = FeatureDetectionNetwork(network_name=instance_name,
                                      image_size=args.image_size,
                                      nb_channels=3,
                                      nb_labels=nb_labels,
                                      learning_rate=args.learning_rate,
                                      architecture=args.network)
        loss_function = "binary_crossentropy"
    elif args.model == "semantic_segmentation":
        net = SemanticSegmentationNetwork(network_name=instance_name,
                                          image_size=args.image_size,
                                          nb_channels=nb_labels,
                                          nb_labels=len(train_config["labels"]),
                                          learning_rate=args.learning_rate,
                                          architecture=args.network)
        loss_function = "categorical_crossentropy"
    else:
        utils.logger.error(("Unrecognized model. Please enter 'feature_detection' "
                            "or 'semantic_segmentation'."))
        sys.exit(1)
    model = Model(net.X, net.Y)
    model.compile(loss=loss_function,
                  optimizer='adam',
                  metrics=['acc', 'mae'])
    model.summary()

    # Model training
    STEPS = args.nb_training_image // args.batch_size
    VAL_STEPS = args.nb_validation_image // args.batch_size
    TEST_STEPS = args.nb_testing_image // args.batch_size
    hist = model.fit_generator(train_generator,
                               epochs=args.nb_epochs,
                               steps_per_epoch=STEPS,
                               validation_data=validation_generator,
                               validation_steps=VAL_STEPS)
    metrics = {"epoch": hist.epoch,
               "metrics": hist.history,
               "params": hist.params}
    utils.logger.info("History:\n")
    print(metrics["metrics"])

    score = model.predict_generator(test_generator, steps=TEST_STEPS)
    utils.logger.info("Extract of prediction score:\n")
    print(score[:10])

    test_label_popularity = np.round(score).astype(np.uint8).sum(axis=0) / score.shape[0]
    utils.logger.info("Test label popularity:\n")
    print(test_label_popularity)

    backend.clear_session()
