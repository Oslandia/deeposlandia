
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
import keras.backend as K

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

def init_model(problem, instance_name, image_size, nb_labels, dropout, network):
    """Initialize a convolutional neural network with Keras API starting from a set of parameters

    Parameters
    ----------
    problem : str
        Type of solved problem, either `feature_detection` or `semantic_segmentation`
    instance_name : str
        Name of the instance, for identification purpose
    image_size : int
        Image size, in pixels (height=width)
    nb_labels : int
        Number of output labels
    dropout : float
        Dropout rate
    network : str
        Network architecture

    Returns
    -------
    keras.models.Model
        Convolutional neural network
    """
    K.clear_session()
    if problem == "feature_detection":
        net = FeatureDetectionNetwork(network_name=instance_name,
                                      image_size=image_size,
                                      nb_labels=nb_labels,
                                      dropout=dropout,
                                      architecture=network)
    elif problem == "semantic_segmentation":
        net = SemanticSegmentationNetwork(network_name=instance_name,
                                          image_size=image_size,
                                          nb_labels=nb_labels,
                                          dropout=dropout,
                                          architecture=network)
    else:
        utils.logger.error(("Unrecognized model. Please enter 'feature_detection' "
                            "or 'semantic_segmentation'."))
        sys.exit(1)
    return Model(net.X, net.Y)

def predict(filenames, dataset, problem, datapath="./data", aggregate=False,
            name=None, network=None, batch_size=None, dropout=None,
            learning_rate=None, learning_rate_decay=None,
            output_dir="/tmp/deeposlandia/predicted"):
    """Make label prediction on image indicated by ̀filename`, according to
    considered `problem`

    Parameters
    ----------
    filenames : str
        Name of the image files on the file system
    dataset : str
        Name of the dataset, either `shapes` or `mapillary`
    problem : str
        Name of the considered model, either `feature_detection` or
    `semantic_segmentation`
    datapath : str
        Relative path of dataset repository
    aggregate : bool
        Either or not the labels are aggregated
    name : str
        Name of the saved network
    network : str
        Name of the chosen architecture, either `simple`, `vgg` or `inception`
    batch_size : integer
        Batch size used for training the model
    dropout : float
        Dropout rate used for training the model
    learning_rate : float
        Learning rate used for training the model
    learning_rate_decay : float
        Learning rate decay used for training the model
    output_dir : str
        Path of the output directory, where labelled images will be stored
    (useful only if `problem=semantic_segmentation`)

    Returns
    -------
    dict
        Double predictions (between 0 and 1, acts as percentages) regarding
    each labels

    """
    # `image_paths` is first got as
    # [[image1, ..., image_i], [image_j, ..., image_n]]
    image_paths = [glob.glob(f) for f in filenames]
    # then it is flattened to get a simple list
    flattened_image_paths = sum(image_paths, [])
    images = extract_images(flattened_image_paths)
    image_size = images.shape[1]

    aggregate_value = "full" if not aggregate else "aggregated"
    instance_args = [name, image_size, network, batch_size, aggregate_value,
                     dropout, learning_rate, learning_rate_decay]
    instance_name = utils.list_to_str(instance_args, "_")

    prepro_folder = utils.prepare_preprocessed_folder(datapath,
                                                      dataset,
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

    if any([arg is None for arg in instance_args]):
        utils.logger.info(("Some arguments are None, "
                           "the best model is considered."))
        output_folder = utils.prepare_output_folder(datapath,
                                                    dataset,
                                                    problem)
        instance_filename = ("best-instance-" + str(image_size)
                             + "-" + aggregate_value + ".json")
        instance_path = os.path.join(output_folder, instance_filename)
        dropout, network = utils.recover_instance(instance_path)
        model = init_model(problem, instance_name, image_size, nb_labels, dropout, network)
        checkpoint_filename = ("best-model-" + str(image_size)
                               + "-" + aggregate_value + ".h5")
        checkpoint_full_path = os.path.join(output_folder, checkpoint_filename)
        if os.path.isfile(checkpoint_full_path):
            utils.logger.info("Checkpoint full path : {}".format(checkpoint_full_path))
            model.load_weights(checkpoint_full_path)
            utils.logger.info(("Model weights have been recovered from {}"
                               "").format(checkpoint_full_path))
        else:
            utils.logger.info(("No available trained model for this image size"
                               " with optimized hyperparameters. The "
                               "inference will be done on an untrained model"))
    else:
        utils.logger.info("All instance arguments are filled out.")
        output_folder = utils.prepare_output_folder(datapath,
                                                    dataset,
                                                    problem,
                                                    instance_name)
        model = init_model(problem, instance_name, image_size,
                           nb_labels, dropout, network)
        checkpoints = [item for item in os.listdir(output_folder)
                       if "checkpoint-epoch" in item]
        if len(checkpoints) > 0:
            model_checkpoint = max(checkpoints)
            checkpoint_full_path = os.path.join(output_folder, model_checkpoint)
            model.load_weights(checkpoint_full_path)
            utils.logger.info(("Model weights have been recovered from {}"
                               "").format(checkpoint_full_path))
        else:
            utils.logger.info(("No available checkpoint for this configuration. "
                               "The model will be trained from scratch."))

    y_raw_pred = model.predict(images)

    result = {}
    if problem == "feature_detection":
        label_info = [(i['category'], utils.RGBToHTMLColor(i['color']))
                      for i in train_config['labels']]
        for filename, prediction in zip(flattened_image_paths, y_raw_pred):
            result[filename] = {i[0]: {"probability": 100*round(float(j), 2),
                                       "color": i[1]}
                                for i, j in zip(label_info, prediction)}
        return result
    elif problem == "semantic_segmentation":
        os.makedirs(output_dir, exist_ok=True)
        predicted_labels = np.argmax(y_raw_pred, axis=3)
        encountered_labels = np.unique(predicted_labels)
        meaningful_labels = [x for i, x in enumerate(train_config["labels"])
                         if i in encountered_labels]
        labelled_images = np.zeros(shape=np.append(predicted_labels.shape, 3),
                                   dtype=np.int8)
        for i in range(nb_labels):
            labelled_images[predicted_labels == i] = train_config["labels"][i]["color"]
        for predicted_labels, filename in zip(labelled_images, flattened_image_paths):
            predicted_image = Image.fromarray(predicted_labels, 'RGB')
            predicted_image_path = os.path.join(output_dir,
                                                os.path.basename(filename))
            predicted_image.save(predicted_image_path)
            result[filename] = os.path.basename(filename)
        return {'labels': summarize_config(meaningful_labels),
                'label_images': result}
    else:
        utils.logger.error(("Unknown model argument. Please use "
                            "'feature_detection' or 'semantic_segmentation'."))
        sys.exit(1)

def summarize_config(config):
    """Extract and reshape dataset configuration information in a HTML-printing
    context

    Parameters
    ----------
    config : dict
        Dataset label configuration

    Returns
    -------
    dict
        Simplified dataset configuration for HTML-printing purpose
    """
    return {c['category']: utils.RGBToHTMLColor(c['color']) for c in config}

def extract_images(image_paths):
    """Convert a list of image filenames into a numpy array that contains the
    image data

    Parameters
    ----------
    image_paths : str
        Name of the image files onto the file system

    Returns
    -------
    np.array
        Data that is contained into the image
    """
    x_test = []
    for image_path in image_paths:
        image = Image.open(image_path)
        if image.size[0] != image.size[1]:
            utils.logger.error(("One of the parsed images "
                                "has non-squared dimensions."))
            sys.exit(1)
        x_test.append(np.array(image))
    return np.array(x_test)

if __name__ == '__main__':

    program_description = ("Infer labels on one (or more) image file(s) "
                           "from a trained deep neural network")
    parser = argparse.ArgumentParser(description=program_description)
    parser = add_program_arguments(parser)
    parser = add_instance_arguments(parser)
    args = parser.parse_args()

    y_raw_pred = predict(args.image_paths, args.dataset, args.model, args.datapath,
                         args.aggregate_label, args.name, args.network,
                         args.batch_size, args.dropout,
                         args.learning_rate, args.learning_rate_decay)

    utils.logger.info(y_raw_pred)
