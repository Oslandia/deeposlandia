"""Main method to train neural networks with Keras API
"""

import argparse
from datetime import datetime
import itertools
import json
import os

import daiquiri
import numpy as np

from keras import backend, callbacks
from keras.models import Model
from keras.optimizers import Adam

from deeposlandia import generator, utils
from deeposlandia import AVAILABLE_MODELS
from deeposlandia.datasets import AVAILABLE_DATASETS
from deeposlandia.feature_detection import FeatureDetectionNetwork
from deeposlandia.semantic_segmentation import SemanticSegmentationNetwork
from deeposlandia.metrics import iou, dice_coef

SEED = int(datetime.now().timestamp())


logger = daiquiri.getLogger(__name__)


def get_data(folders, dataset, model, image_size, batch_size):
    """On the file system, recover `dataset` that can solve `model` problem

    Parameters
    ----------
    folders : dict
        Dictionary of useful folders that indicates paths to data
    dataset : str
        Name of the used dataset (*e.g.* `shapes` or `mapillary`)
    model : str
        Name of the addressed research problem (*e.g.* `feature_detection` or
    `semantic_segmentation`)
    image_size : int
        Size of the images, in pixel (height=width)
    batch_size : int
        Number of images in each batch

    Returns
    -------
    tuple
        Number of labels in the dataset, as well as training and validation
    data generators

    """
    # Data gathering
    if os.path.isfile(folders["training_config"]):
        train_config = utils.read_config(folders["training_config"])
        label_ids = [
            x["id"] for x in train_config["labels"] if x["is_evaluate"]
        ]
        train_generator = generator.create_generator(
            dataset,
            model,
            folders["training"],
            image_size,
            batch_size,
            train_config["labels"],
            seed=SEED,
        )
    else:
        raise FileNotFoundError(
            "There is no training data with the given parameters. Please "
            "generate a valid dataset before calling the training program."
        )
    if os.path.isfile(folders["validation_config"]):
        validation_generator = generator.create_generator(
            dataset,
            model,
            folders["validation"],
            image_size,
            batch_size,
            train_config["labels"],
            seed=SEED,
        )
    else:
        raise FileNotFoundError(
            "There is no validation data with the given parameters. Please "
            "generate a valid dataset before calling the training program."
        )
    nb_labels = len(label_ids)
    return nb_labels, train_generator, validation_generator


def run_model(
    train_generator,
    validation_generator,
    dl_model,
    output_folder,
    instance_name,
    image_size,
    nb_labels,
    nb_epochs,
    nb_training_image,
    nb_validation_image,
    batch_size,
    dropout,
    network,
    learning_rate,
    learning_rate_decay,
):
    """Run deep learning `dl_model` starting from training and validation data
    generators, depending on a range of hyperparameters

    Parameters
    ----------
    train_generator : generator
        Training data generator
    validation_generator : generator
        Validation data generator
    dl_model : str
        Name of the addressed research problem (*e.g.* `feature_detection` or
    `semantic_segmentation`)
    output_folder : str
        Name of the folder where the trained model will be stored on the file
    system
    instance_name : str
        Name of the instance
    image_size : int
        Size of images, in pixel (height=width)
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
        Dictionary that summarizes the instance and the corresponding model
    performance (measured by validation accuracy)
    """
    if dl_model == "featdet":
        net = FeatureDetectionNetwork(
            network_name=instance_name,
            image_size=image_size,
            nb_channels=3,
            nb_labels=nb_labels,
            architecture=network,
        )
        loss_function = "binary_crossentropy"
    elif dl_model == "semseg":
        net = SemanticSegmentationNetwork(
            network_name=instance_name,
            image_size=image_size,
            nb_channels=3,
            nb_labels=nb_labels,
            architecture=network,
        )
        loss_function = "categorical_crossentropy"
    else:
        raise ValueError(
            f"Unrecognized model: {dl_model}. Please choose amongst {AVAILABLE_MODELS}"
        )
    model = Model(net.X, net.Y)
    opt = Adam(lr=learning_rate, decay=learning_rate_decay)
    metrics = ["acc", iou, dice_coef]
    model.compile(loss=loss_function, optimizer=opt, metrics=metrics)

    # Model training
    steps = max(nb_training_image // batch_size, 1)
    val_steps = max(nb_validation_image // batch_size, 1)

    checkpoint_files = [
        item
        for item in os.listdir(output_folder)
        if "checkpoint-epoch" in item
    ]
    if len(checkpoint_files) > 0:
        model_checkpoint = max(checkpoint_files)
        trained_model_epoch = int(model_checkpoint[-5:-3])
        checkpoint_complete_path = os.path.join(
            output_folder, model_checkpoint
        )
        model.load_weights(checkpoint_complete_path)
        logger.info(
            "Model weights have been recovered from %s",
            checkpoint_complete_path,
        )
    else:
        logger.info(
            (
                "No available checkpoint for this configuration. "
                "The model will be trained from scratch."
            )
        )
        trained_model_epoch = 0

    checkpoint_filename = os.path.join(
        output_folder, "checkpoint-epoch-{epoch:03d}.h5"
    )
    checkpoint = callbacks.ModelCheckpoint(
        checkpoint_filename,
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        period=1,
    )
    terminate_on_nan = callbacks.TerminateOnNaN()
    earlystop = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, verbose=1, mode="max"
    )
    csv_logger = callbacks.CSVLogger(
        os.path.join(output_folder, "training_metrics.csv"), append=True
    )

    hist = model.fit_generator(
        train_generator,
        epochs=nb_epochs,
        initial_epoch=trained_model_epoch,
        steps_per_epoch=steps,
        validation_data=validation_generator,
        validation_steps=val_steps,
        callbacks=[checkpoint, earlystop, terminate_on_nan, csv_logger],
    )
    ref_metric = max(hist.history.get("val_acc", [np.nan]))
    return {
        "model": model,
        "val_acc": ref_metric,
        "batch_size": batch_size,
        "network": network,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "learning_rate_decay": learning_rate_decay,
    }


def main(args):
    # Grid search
    model_output = []
    output_folder = utils.prepare_output_folder(
        args.datapath, args.dataset, args.image_size, args.model
    )
    for batch_size in args.batch_size:
        logger.info("Generating data with batch of %s images...", batch_size)
        # Data generator building
        prepro_folder = utils.prepare_preprocessed_folder(
            args.datapath, args.dataset, args.image_size
        )
        nb_labels, train_gen, valid_gen = get_data(
            prepro_folder,
            args.dataset,
            args.model,
            args.image_size,
            batch_size,
        )
        for parameters in itertools.product(
            args.dropout,
            args.network,
            args.learning_rate,
            args.learning_rate_decay,
        ):
            logger.info("Instance: %s", utils.list_to_str(parameters))
            # Data path and repository management
            dropout, network, learning_rate, learning_rate_decay = parameters
            instance_args = [
                args.name,
                args.image_size,
                network,
                batch_size,
                dropout,
                learning_rate,
                learning_rate_decay,
            ]
            instance_name = utils.list_to_str(instance_args, "_")
            instance_folder = os.path.join(output_folder["checkpoints"], instance_name)
            # Model running
            model_output.append(
                run_model(
                    train_gen,
                    valid_gen,
                    args.model,
                    instance_folder,
                    instance_name,
                    args.image_size,
                    nb_labels,
                    args.nb_epochs,
                    args.nb_training_image,
                    args.nb_validation_image,
                    batch_size,
                    *parameters
                )
            )
            logger.info("Instance result: %s", model_output[-1])

    # Recover best instance starting from validation accuracy
    best_instance = max(model_output, key=lambda x: x["val_acc"])

    # Save best model
    best_instance["model"].save(output_folder["best-model"])
    with open(output_folder["best-instance"], "w") as fobj:
        json.dump(
            {
                key: best_instance[key]
                for key in best_instance
                if key != "model"
            },
            fobj,
        )

    backend.clear_session()
