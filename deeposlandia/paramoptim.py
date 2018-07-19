"""Main method to train neural networks with Keras API
"""

import argparse
import itertools
import json
import numpy as np
import os
import sys

from datetime import datetime

from keras import backend, callbacks
from keras.models import Model
from keras.optimizers import Adam

from deeposlandia import generator, utils
from deeposlandia.feature_detection import FeatureDetectionNetwork
from deeposlandia.semantic_segmentation import SemanticSegmentationNetwork

SEED = int(datetime.now().timestamp())


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
    parser.add_argument('-p', '--datapath',
                        default="./data",
                        help="Relative path towards data directory")
    parser.add_argument('-s', '--image-size',
                        default=224,
                        type=int,
                        help=("Desired size of images (width = height)"))
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
                        type=int, nargs='+',
                        default=[20, 50, 100],
                        help=("Number of images that must be contained "
                              "into a single batch"))
    parser.add_argument('-d', '--dropout',
                        type=float, nargs='+',
                        default=[0.5, 0.75, 1.0],
                        help=("Percentage of dropped out neurons "
                              "during training"))
    parser.add_argument('-L', '--learning-rate',
                        default=[0.01, 0.001, 0.0001],
                        type=float, nargs="+",
                        help=("List of learning rate components (starting LR, "
                              "decay steps and decay rate)"))
    parser.add_argument('-l', '--learning-rate-decay',
                        default=[0.001, 0.0001, 0.00001],
                        type=float, nargs="+",
                        help=("List of learning rate components (starting LR, "
                              "decay steps and decay rate)"))
    parser.add_argument('-N', '--network', nargs='+',
                        default=['simple', 'vgg16'],
                        help=("Neural network size, either 'simple', 'vgg' or "
                              "'inception' ('simple' refers to 3 conv/pool "
                              "blocks and 1 fully-connected layer; the others "
                              "refer to state-of-the-art networks)"))
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
    parser.add_argument('-e', '--nb-epochs',
                        type=int,
                        default=0,
                        help=("Number of training epochs (one epoch means "
                              "scanning each training image once)"))
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


def get_data(folders, dataset, model, image_size, batch_size):
    """On the file system, recover `dataset` that can solve `model` problem

    Parameters
    ----------
    folders : dict
        Dictionary of useful folders that indicates paths to data
    dataset : str
        Name of the used dataset (*e.g.* `shapes` or `mapillary`)
    model : str
        Name of the addressed research problem (*e.g.* `feature_detection` or `semantic_segmentation`)
    image_size : int
        Size of the images, in pixel (height=width)
    batch_size : int
        Number of images in each batch

    Returns
    -------
    tuple
        Number of labels in the dataset, as well as training, validation and testing data generators

    """
    # Data gathering
    if (os.path.isfile(folders["training_config"])
        and os.path.isfile(folders["validation_config"])
        and os.path.isfile(folders["testing_config"])):
        train_config = utils.read_config(folders["training_config"])
        label_ids = [x['id'] for x in train_config['labels'] if x['is_evaluate']]
        train_generator = generator.create_generator(
            dataset,
            model,
            folders["training"],
            image_size,
            batch_size,
            train_config["labels"],
            seed=SEED)
        validation_generator = generator.create_generator(
            dataset,
            model,
            folders["validation"],
            image_size,
            batch_size,
            train_config["labels"],
            seed=SEED)
        test_generator = generator.create_generator(
            dataset,
            model,
            folders["testing"],
            image_size,
            batch_size,
            train_config["labels"],
            inference=True,
            seed=SEED)
    else:
        utils.logger.error(("There is no valid data with the specified parameters. "
                           "Please generate a valid dataset before calling the training program."))
        sys.exit(1)
    nb_labels = len(label_ids)
    return nb_labels, train_generator, validation_generator, test_generator


def run_model(train_generator, validation_generator, dl_model, output_folder,
              instance_name, image_size, aggregate_value, nb_labels, nb_epochs,
              nb_training_image, nb_validation_image,
              batch_size, dropout, network, learning_rate, learning_rate_decay):
    """Run deep learning `dl_model` starting from training and validation data generators, depending on a
              range of hyperparameters

    Parameters
    ----------
    train_generator : generator
        Training data generator
    validation_generator : generator
        Validation data generator
    dl_model : str
        Name of the addressed research problem (*e.g.* `feature_detection` or `semantic_segmentation`)
    output_folder : str
        Name of the folder where the trained model will be stored on the file system
    instance_name : str
        Name of the instance
    image_size : int
        Size of images, in pixel (height=width)
    aggregate_value : str
        Label aggregation policy (either `full` or `aggregated`)
    nb_labels : int
        Number of labels into the dataset
    nb_epochs : int
        Number of epochs during which models will be trained
    nb_training_image : int
        Number of images into the training dataset
    nb_validation_image : int
        Number of images into the validation dataset
    batch_size : int
        Number of images into each batch
    dropout : float
        Probability of keeping a neuron during dropout phase
    network : str
        Neural network architecture (*e.g.* `simple`, `vgg`, `inception`)
    learning_rate : float
        Starting learning rate
    learning_rate_decay : float
        Learning rate decay

    Returns
    -------
    dict
        Dictionary that summarizes the instance and the corresponding model performance (measured
    by validation accuracy)
    """
    if dl_model == "feature_detection":
        net = FeatureDetectionNetwork(network_name=instance_name,
                                      image_size=image_size,
                                      nb_channels=3,
                                      nb_labels=nb_labels,
                                      architecture=network)
        loss_function = "binary_crossentropy"
    elif dl_model == "semantic_segmentation":
        net = SemanticSegmentationNetwork(network_name=instance_name,
                                          image_size=image_size,
                                          nb_channels=3,
                                          nb_labels=nb_labels,
                                          architecture=network)
        loss_function = "categorical_crossentropy"
    else:
        utils.logger.error(("Unrecognized model. Please enter 'feature_detection' "
                            "or 'semantic_segmentation'."))
        sys.exit(1)
    model = Model(net.X, net.Y)
    opt = Adam(lr=learning_rate, decay=learning_rate_decay)
    model.compile(loss=loss_function, optimizer=opt, metrics=['acc'])

    # Model training
    steps = nb_training_image // batch_size
    val_steps = nb_validation_image // batch_size


    checkpoints = [item for item in os.listdir(output_folder)
                   if os.path.isfile(os.path.join(output_folder, item))]
    if len(checkpoints) > 0:
        model_checkpoint = max(checkpoints)
        trained_model_epoch = int(model_checkpoint[-5:-3])
        checkpoint_complete_path = os.path.join(output_folder, model_checkpoint)
        model.load_weights(checkpoint_complete_path)
        utils.logger.info(("Model weights have been recovered from {}"
                           "").format(checkpoint_complete_path))
    else:
        utils.logger.info(("No available checkpoint for this configuration. "
                           "The model will be trained from scratch."))
        trained_model_epoch = 0


    checkpoint_filename = os.path.join(output_folder,
                                       "checkpoint-epoch-{epoch:03d}.h5")
    checkpoints = callbacks.ModelCheckpoint(
        checkpoint_filename,
        monitor='val_acc',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto', period=1)
    terminate_on_nan = callbacks.TerminateOnNaN()
    earlystop = callbacks.EarlyStopping(monitor='val_acc',
                                        min_delta=0.001,
                                        patience=10,
                                        verbose=1,
                                        mode='auto')
    hist = model.fit_generator(train_generator,
                               epochs=nb_epochs,
                               initial_epoch=trained_model_epoch,
                               steps_per_epoch=steps,
                               validation_data=validation_generator,
                               validation_steps=val_steps,
                               callbacks=[checkpoints, earlystop, terminate_on_nan])
    ref_metric = max(hist.history.get("val_acc", [np.nan]))
    return {'model': model, 'val_acc': ref_metric,
            'batch_size': batch_size, 'network': network, 'dropout': dropout,
            'learning_rate': learning_rate, 'learning_rate_decay': learning_rate_decay}


if __name__ == '__main__':

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description=("Convolutional Neural Netw"
                                                  "ork on street-scene images:"
                                                  " hyper-parameter "
                                                  "optimization"))
    parser = add_instance_arguments(parser)
    parser = add_hyperparameters(parser)
    parser = add_training_arguments(parser)
    args = parser.parse_args()

    aggregate_value = "full" if not args.aggregate_label else "aggregated"

    # Grid search
    model_output = []
    for batch_size in args.batch_size:
        utils.logger.info(("Generating data with batch of {} images..."
                           "").format(batch_size))
        # Data generator building
        prepro_folder = utils.prepare_preprocessed_folder(args.datapath,
                                                          args.dataset,
                                                          args.image_size,
                                                          aggregate_value)
        nb_labels, train_gen, valid_gen, test_gen = get_data(prepro_folder,
                                                             args.dataset,
                                                             args.model,
                                                             args.image_size,
                                                             batch_size)
        for parameters in itertools.product(args.dropout,
                                            args.network,
                                            args.learning_rate,
                                            args.learning_rate_decay):
            utils.logger.info(utils.list_to_str(parameters))
            # Data path and repository management
            dropout, network, learning_rate, learning_rate_decay = parameters
            instance_args = [args.name, args.image_size, network,
                             batch_size, aggregate_value, dropout,
                             learning_rate, learning_rate_decay]
            instance_name = utils.list_to_str(instance_args, "_")
            output_folder = utils.prepare_output_folder(args.datapath,
                                                        args.dataset,
                                                        args.model,
                                                        instance_name)
            # Model running
            model_output.append(run_model(train_gen, valid_gen, args.model,
                                          output_folder, instance_name,
                                          args.image_size, aggregate_value,
                                          nb_labels, args.nb_epochs,
                                          args.nb_training_image,
                                          args.nb_validation_image, batch_size,
                                          *parameters))
            utils.logger.info("Instance result: {}".format(model_output[-1]))

    # Recover best instance starting from validation accuracy
    best_instance = max(model_output, key=lambda x: x['val_acc'])

    # Save best model
    output_folder = utils.prepare_output_folder(args.datapath,
                                                args.dataset,
                                                args.model)
    instance_name = os.path.join(output_folder,
                                 "best-{}-" + str(args.image_size)
                                 + "-" + aggregate_value + ".{}")
    best_instance["model"].save(instance_name.format("model", "h5"))
    with open(instance_name.format("instance", "json"), "w") as fobj:
              json.dump({key:best_instance[key]
                         for key in best_instance if key != 'model'}, fobj)

    backend.clear_session()
