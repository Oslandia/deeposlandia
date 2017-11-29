#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
/**
 *   Raphael Delhome - september 2017
 *
 *   This library is free software; you can redistribute it and/or
 *   modify it under the terms of the GNU Library General Public
 *   License as published by the Free Software Foundation; either
 *   version 2 of the License, or (at your option) any later version.
 *   
 *   This library is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *   Library General Public License for more details.
 *   You should have received a copy of the GNU Library General Public
 *   License along with this library; if not, see <http://www.gnu.org/licenses/>
 */
"""

# This script aims to train a neural network model in order to read street
# scene images produced by Mapillary (https://www.mapillary.com/dataset/vistas)

import argparse
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import sys
import time

import cnn_layers
import dashboard_building
import glossary_reading as gr
import utils

def run(nbconv, nbfullyconn, nb_epochs, nb_iter, mode, label_list,
        image_size, weight_policy, start_lr, decay_steps, decay_rate,
        name, datapath):
    """Train and/or test a convolutional neural network on street-scene images
    to detect features

    Parameters
    ----------
    nbconv: integer
        Number of convolutional layer to setup in the network
    nbfullconn: integer
        Number of fully connected layer to setup in the network
    nb_epochs: integer
        Number of training epoch(s)
    nb_iter: integer
        Number of training iterations, if None, the model run during nb_epochs
    * nb_batches iterations
    mode: object
        String designing the running mode ("train", "test", or "both")
    label_list: list
        List of label indices (integers) that will be considered during training
    weight_policy: object
        String designing the way label loss contributions are weighted ("base",
    "global", "batch", "centeredglobal", "centeredbatch")
    start_lr: integer
        Starting learning rate (between 0 and 1) - cf Tensorflow tf.train.exponential_decay
    decay_steps: double
        Step normalization term - cf Tensorflow tf.train.exponential_decay
    decay_rate: double
        Learning rate decay (between 0 and 1) - cf Tensorflow tf.train.exponential_decay
    name: object
        String designing the name of the network
    datapath: object
        String designing the relative path to dataset directory
    
    """
    NETWORK_NAME = (name + "_" + weight_policy + "_"
                    + str(nbconv) + "_" + str(nbfullyconn))
    if mode == "train":
        utils.logger.info("Model {} training".format(NETWORK_NAME))
    elif mode == "test":
        utils.logger.info("Model {} testing".format(NETWORK_NAME))
    elif mode == "both":
        utils.logger.info("Model {} training and testing".format(NETWORK_NAME))
    config_file_name = os.path.join("..", "models", NETWORK_NAME + ".json")
    with open(config_file_name) as config_file:
        cnn_hyperparam = json.load(config_file)

    # image dimensions (width, height, number of channels)
    IMAGE_WIDTH   = image_size[0]
    IMAGE_HEIGHT  = image_size[1]
    NUM_CHANNELS  = 3 # Colored images (RGB)
    NETWORK_NAME = (NETWORK_NAME + "_" + str(image_size[0])
                    + "_" + str(image_size[1]))

    # number of output classes
    N_CLASSES = len(label_list)
    # number of images per batch
    BATCH_SIZE = 20
    N_IMAGES = len(os.listdir(os.path.join(datapath, "training",
                                           "input" + "_" + str(image_size[0])
                                           + "_" + str(image_size[1]))))
    N_VAL_IMAGES = len(os.listdir(os.path.join(datapath, "validation",
                                           "input" + "_" + str(image_size[0])
                                           + "_" + str(image_size[1]))))
    N_BATCHES = int(N_IMAGES / BATCH_SIZE)
    N_VAL_BATCHES = int(N_VAL_IMAGES / BATCH_SIZE)
    # learning rate tuning (exponential decay)
    START_LR = start_lr
    DECAY_STEPS = decay_steps
    DECAY_RATE = decay_rate
    # percentage of nodes that are briefly removed during training process
    DROPOUT = 2/3.0
    # number of epochs (one epoch = all images have been used for training)
    N_EPOCHS = nb_epochs
    # printing frequency during training
    SKIP_STEP = 10

    # Data recovering
    train_image_batch, train_label_batch, train_filename_batch = \
    cnn_layers.prepare_data(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS,
                            BATCH_SIZE, label_list, datapath,
                            "training", "training_data_pipe")
    val_image_batch, val_label_batch, val_filename_batch = \
    cnn_layers.prepare_data(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS,
                            BATCH_SIZE, label_list, datapath,
                            "validation", "valid_data_pipe")

    # Definition of TensorFlow placeholders
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH,
                                    NUM_CHANNELS], name='X')
    Y = tf.placeholder(tf.float32, [None, N_CLASSES], name='Y')
    dropout = tf.placeholder(tf.float32, name='dropout')
    class_w = tf.placeholder(tf.float32, [N_CLASSES],
                             name='weights_per_label')

    # Model building
    logits, y_raw_pred, y_pred = cnn_layers.convnet_building(X, cnn_hyperparam,
                                                             image_size[0],
                                                             image_size[1],
                                                             NUM_CHANNELS,
                                                             N_CLASSES,
                                                             dropout,
                                                             NETWORK_NAME,
                                                             nbconv,
                                                             nbfullyconn)

    # Loss function design
    output = cnn_layers.define_loss(Y, logits, y_raw_pred, class_w, START_LR,
                                    DECAY_STEPS, DECAY_RATE, NETWORK_NAME)

    # Running the neural network
    with tf.Session() as sess:
        # Initialize the tensorflow variables
        sess.run(tf.global_variables_initializer())
        # Declare a saver instance and a summary writer to store the network
        saver = tf.train.Saver(max_to_keep=1)
        graph_path = os.path.join(datapath, 'graphs', NETWORK_NAME)
        writer = tf.summary.FileWriter(graph_path, sess.graph)
        
        # Create folders to store checkpoints
        ckpt_path = os.path.join(datapath, 'checkpoints', NETWORK_NAME)
        utils.make_dir(os.path.dirname(ckpt_path))
        utils.make_dir(ckpt_path)
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        # If that checkpoint exists, restore from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            utils.logger.info(("Recover model state "
                               "from {}").format(ckpt.model_checkpoint_path))
        initial_step = output["gs"].eval(session=sess)

        # Initialize threads to begin batching operations
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        if mode in ["train", "both"]:
            # Train the model
            start_time = time.time()
            dashboard = []
            val_dashboard = []
            if weight_policy == "base":
                w_batch = np.repeat(1.0, N_CLASSES)
            elif weight_policy == "global":
                label_counter = gr.count_image_per_label(datapath, image_size)
                label_counter = [label_counter[l] for l in label_list]
                w_batch = utils.compute_monotonic_weights(N_IMAGES,
                                                          label_counter)
            elif weight_policy == "centeredglobal":
                label_counter = gr.count_image_per_label(datapath, image_size)
                label_counter = [label_counter[l] for l in label_list]
                w_batch = utils.compute_centered_weights(N_IMAGES, label_counter)

            if nb_iter is None:
                nb_iter = N_BATCHES * N_EPOCHS
            for index in range(initial_step, nb_iter):
                X_batch, Y_batch = sess.run([train_image_batch,
                                             train_label_batch])
                if weight_policy == "batch":
                    label_counter = [sum(s) for s in np.transpose(Y_batch)]
                    w_batch = utils.compute_monotonic_weights(BATCH_SIZE,
                                                              label_counter)
                elif weight_policy == "centeredbatch":
                    label_counter = [sum(s) for s in np.transpose(Y_batch)]
                    w_batch = utils.compute_centered_weights(BATCH_SIZE,
                                                             label_counter)
                fd = {X: X_batch, Y: Y_batch, dropout: 1.0, class_w: w_batch}

                if (index + 1) % SKIP_STEP == 0 or index == initial_step:
                    Y_pred, loss, bpmll, lr = sess.run([y_pred,
                                                        output["loss"],
                                                        output["bpmll"],
                                                        output["lrate"]],
                                                       feed_dict=fd)
                    db_batch = dashboard_building.dashboard_building(Y_batch,
                                                                     Y_pred)
                    db_batch.insert(0, bpmll)
                    db_batch.insert(0, loss)
                    db_batch.insert(0, index)
                    dashboard.append(db_batch)

                    if mode == "both":
                        # Run the model on validation dataset
                        partial_val_dashboard = []
                        for val_index in range(N_VAL_BATCHES):
                            X_val_batch, Y_val_batch = \
                            sess.run([val_image_batch, val_label_batch])
                            fd_val = {X: X_val_batch, Y: Y_val_batch,
                                      dropout: 1.0, class_w: w_batch}
                            Y_pred_val, loss_batch_val, bpmll_val =\
                    sess.run([y_pred, output["loss"], output["bpmll"]],
                            feed_dict=fd_val)
                            db_val_batch = \
                            dashboard_building.dashboard_building(Y_val_batch,
                                                                  Y_pred_val)
                            db_val_batch.insert(0, bpmll_val)
                            db_val_batch.insert(0, loss_batch_val)
                            db_val_batch.insert(0, index)
                            partial_val_dashboard.append(db_val_batch)
                        curval_dashboard = (pd.DataFrame(partial_val_dashboard)
                                            .apply(lambda x: x.mean(), axis=0))
                        curval_dashboard = list(curval_dashboard)
                        val_dashboard.append(curval_dashboard)
                        utils.logger.info(("Step {} (lr={:1.3f}): loss="
                                           "{:5.3f}, accuracy={:1.3f} "
                                           "(validation: {:1.3f}), precision="
                                           "{:1.3f}, recall={:1.3f}")
                                          .format(index, lr, loss,
                                                  db_batch[8],
                                                  db_val_batch[8],
                                                  db_batch[9], db_batch[10]))
                    else:
                        utils.logger.info(("Step {} (lr={:1.3f}): loss={:5.3f}"
                                           ", accuracy={:1.3f}, precision="
                                           "{:1.3f}, recall={:1.3f}")
                                          .format(index, lr, loss,
                                                  db_batch[8], db_batch[9],
                                                  db_batch[10]))

                # Run the model to do a new training iteration
                fd = {X: X_batch, Y: Y_batch, dropout: DROPOUT, class_w: w_batch}
                sess.run(output["optim"], feed_dict=fd)

                # If all training batches have been scanned, save the model
                if (index + 1) % N_BATCHES == 0:
                    utils.logger.info(("Checkpoint {}/checkpoints/{}/epoch-{}"
                                       " creation")
                                      .format(datapath, NETWORK_NAME,
                                              index))
                    saver.save(sess, global_step=index,
                               save_path=os.path.join(datapath, 'checkpoints',
                                                      NETWORK_NAME, 'epoch'))

            utils.logger.info("Optimization Finished!")
            utils.logger.info("Total time: {:.2f} seconds".format(time.time() -
                                                                  start_time))

            if initial_step < N_BATCHES * N_EPOCHS:
                # The results are stored as a df and saved on the file system
                db_columns = ["epoch", "loss", "bpmll_loss", "hamming_loss",
                              "true_neg", "false_pos", "false_neg", "true_pos",
                              "accuracy", "precision", "recall", "F_measure"]
                db_columns_by_label = [["tn_"+str(i),
                                        "fp_"+str(i),
                                        "fn_"+str(i),
                                        "tp_"+str(i),
                                        "accuracy_"+str(i),
                                        "precision_"+str(i),
                                        "recall_"+str(i)]
                                       for i in label_list]
                db_columns_by_label = utils.unnest(db_columns_by_label)
                db_columns = db_columns + db_columns_by_label
                param_history = pd.DataFrame(dashboard, columns=db_columns)
                param_history = param_history.set_index("epoch")
                val_param_history = pd.DataFrame(val_dashboard,
                                                 columns=db_columns)
                val_param_history = val_param_history.set_index("epoch")
                utils.make_dir(os.path.join("..", "data", "results"))
                result_file_name = os.path.join(datapath, "results",
                                                NETWORK_NAME + ".csv")
                val_result_file_name = os.path.join(datapath, "results",
                                                    NETWORK_NAME + "_val.csv")
                if initial_step == 0:
                    param_history.to_csv(result_file_name, index=True)
                    if mode in ["both", "test"]:
                        val_param_history.to_csv(val_result_file_name,
                                                 index=True)
                else:
                    param_history.to_csv(result_file_name, mode='a',
                                         index=True, header=False)
                    if mode in ["both", "test"]:
                        val_param_history.to_csv(val_result_file_name, mode='a',
                                                 index=True, header=False)

                # Training results are then saved as a multiplot
                step = output["gs"].eval(session=sess)
                complete_dashboard = pd.read_csv(result_file_name)
                plot_file_name = os.path.join("..", "images",
                                              (NETWORK_NAME + "_s"
                                               + str(step) + ".png"))
                dashboard_building.plot_dashboard(complete_dashboard,
                                                  plot_file_name,
                                                  label_list)
        elif mode == "test":
            utils.logger.info(("Test model after {}"
                               " training steps!").format(initial_step))

            # Run the model on validation dataset
            val_dashboard = []
            for val_index in range(N_VAL_BATCHES):
                X_val_batch, Y_val_batch = sess.run([val_image_batch,
                                                     val_label_batch])
                fd_val = {X: X_val_batch, Y: Y_val_batch, dropout: 1.0,
                          class_w: np.repeat(1.0, N_CLASSES)}
                Y_pred_val = sess.run([y_pred], feed_dict=fd_val)
                db_val_batch = \
                dashboard_building.dashboard_building(Y_val_batch, Y_pred_val[0])
                db_val_batch.insert(0, val_index)
                val_dashboard.append(db_val_batch)
                if (index + 1) % SKIP_STEP == 0 or index == initial_step:
                    utils.logger.info(("Step {}: accuracy = {:1.3f}, precision"
                                       " = {:1.3f}, recall = {:1.3f}")
                                      .format(val_index, db_val_batch[2],
                                              db_val_batch[3], db_val_batch[4]))
            val_cur_dashboard = list(pd.DataFrame(val_dashboard)
                                     .apply(lambda x: x.mean(), axis=0))
            val_dashboard.append(val_cur_dashboard)
            utils.logger.info(("Model validation: accuracy={:1.3f}, precision"
                               "={:1.3f}, recall={:1.3f}")
                              .format(val_cur_dashboard[2],
                                      val_cur_dashboard[3],
                                      val_cur_dashboard[4]))

        # Stop the threads used during the process
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # Manage argument parsing
    parser = argparse.ArgumentParser(description=("Convolutional Neural Netw"
                                                  "ork on street-scene images"))
    parser.add_argument('-c', '--nbconv', required=False, type=int,
                        nargs='?', default=2,
                        help=("The number of convolutional layers "
                              "that must be inserted into the network"))
    parser.add_argument('-d', '--datapath', required=False,
                        default="../data", nargs='?',
                        help="""The relative path towards data directory""")
    parser.add_argument('-e', '--nb-epochs', required=False, type=int,
                        default=5, nargs='?',
                        help=("The number of training epochs (one epoch means "
                              "scanning each training image once)"))
    parser.add_argument('-f', '--nbfullyconn', required=False, type=int,
                        nargs='?', default=1,
                        help=("The number of fully-connected layers "
                              "that must be inserted into the network"))
    parser.add_argument('-g', '--glossary-printing', action="store_true",
                        help=("True if the program must only "
                              "print the glossary, false otherwise)"))
    parser.add_argument('-l', '--label-list', required=False, nargs="+",
                        default=-1, type=int,
                        help=("The list of label indices that "
                              "will be considered during training process"))
    parser.add_argument('-m', '--mode', required=False, default="train",
                        nargs='?', help=("The network running mode"
                                         "('train', 'test', 'both')"))
    parser.add_argument('-n', '--name', default=["cnn_mapil"], nargs='+',
                        help=("The model name that will be used for results, "
                              "checkout and graph storage on file system"))
    parser.add_argument('-p', '--prepare-data', action="store_true",
                        help=("True if the data must be prepared, "
                              "false otherwise"))
    parser.add_argument('-r', '--learning-rate', required=False, nargs="+",
                        default=[0.01, 0.95, 1000], type=int,
                        help=("List of learning rate components (starting LR, "
                              "decay steps and decay rate)"))
    parser.add_argument('-s', '--image-size', nargs="+",
                        default=[512, 384], type=int,
                        help=("The desired size of images (width, height)"))
    parser.add_argument('-tl', '--training-limit', default=None, type=int,
                        help=("Number of training iteration, "
                              "if not specified the model run during "
                              "nb-epochs * nb-batchs iterations"))
    parser.add_argument('-w', '--weights', default=["base"], nargs='+',
                        help=("The weight policy to apply on label "
                              "contributions to loss: either 'base' "
                              "(default case), 'global', 'batch', "
                              "'centeredglobal', 'centeredbatch'"))
    args = parser.parse_args()

    if type(args.image_size) is not list or len(args.image_size) != 2:
        utils.logger.error(("Unsupported image size. Please provide two "
                            "integers (respectively width and height)"))
        sys.exit(1)

    if args.mode not in ["train", "test", "both"]:
        utils.logger.error(("Unsupported running mode. "
                            "Please choose amongst 'train', 'test' or 'both'."))
        sys.exit(1)

    weights = ["base", "global", "batch", "centeredbatch", "centeredglobal"] 
    if sum([w in weights for w in args.weights]) != len(args.weights):
        utils.logger.error(("Unsupported weighting policy. Please choose "
                            "amongst 'base', 'global', 'batch', "
                            "'centeredglobal' or 'centeredbatch'."""))
        utils.logger.info("'base': Regular weighting scheme...")
        utils.logger.info(("'global': Label contributions to loss are "
                           "weighted with respect to label popularity "
                           "within the dataset (decreasing weights)..."))
        utils.logger.info(("'batch': Label contributions to loss are weighted "
                           "with respect to label popularity within the "
                           "dataset (convex weights with min at 50%)..."))
        utils.logger.info(("'centeredbatch': Label contributions to loss are "
                           "weighted with respect to label popularity within "
                           "each batch (decreasing weights)..."))
        utils.logger.info(("'centeredglobal': Label contributions to loss are "
                           "weighted with respect to label popularity within "
                           "each batch (convex weights with min at 50%)..."))
        sys.exit(1)

    mapil_glossary = gr.read_glossary(os.path.join(args.datapath, "config.json"))
    nb_labels = gr.label_quantity(mapil_glossary)
    if args.label_list == -1:
        label_list = [i for i in range(nb_labels)]
    else:
        label_list = args.label_list
        print(label_list)
        if sum([l>=nb_labels for l in args.label_list]) > 0:
            utils.logger.error(("Unsupported label list. "
                                "Please enter a list of integers comprised"
                                "between 0 and {}".format(nb_labels)))
            sys.exit(1)

    if args.glossary_printing:
        glossary_description = gr.build_category_description(mapil_glossary)
        nb_images_per_label = gr.count_image_per_label(args.datapath,
                                                       args.image_size)
        glossary_description["nb_images"] = nb_images_per_label
        utils.logger.info(("Data glossary:\n{}"
                           "").format(glossary_description.iloc[label_list,:]))
        sys.exit(0)

    if len(args.learning_rate) != 3:
        utils.logger.error(("There must be 3 learning rate components "
                            "(start, decay steps and decay rate"
                            "; actually, there is/are {}"
                            "").format(len(args.learning_rate)))
        sys.exit(1)

    if args.prepare_data:
        utils.mapillary_data_preparation(args.datapath, "training",
                                         args.image_size, nb_labels)
        utils.mapillary_data_preparation(args.datapath, "validation",
                                         args.image_size, nb_labels)
    else:
        for n in args.name:
            for w in args.weights:
                run(args.nbconv, args.nbfullyconn, args.nb_epochs,
                    args.training_limit, args.mode, label_list,
                    args.image_size, w, args.learning_rate[0],
                    args.learning_rate[1], args.learning_rate[2],
                    n, args.datapath)
    sys.exit(0)
