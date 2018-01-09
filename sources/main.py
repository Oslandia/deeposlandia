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
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import sys
import time

import cnn_layers
import dashboard_building as db
import glossary_reading as gr
import utils

def run(nbconv, nbfullyconn, nb_epochs, nb_iter, mode, label_list,
        image_size, weight_policy, start_lr, decay_steps, decay_rate,
        drpt, save_step, log_step, batch_size, name, datapath):
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
        List of label indices (integers) that will be considered during
    training
    image_size: integer
        Desired image size (width and height are equals)
    weight_policy: object
        String designing the way label loss contributions are weighted ("base",
    "global", "batch", "centeredglobal", "centeredbatch")
    start_lr: integer
        Starting learning rate (between 0 and 1) - cf Tensorflow tf.train.exponential_decay
    decay_steps: double
        Step normalization term - cf Tensorflow tf.train.exponential_decay
    decay_rate: double
        Learning rate decay (between 0 and 1) - cf Tensorflow
    tf.train.exponential_decay
    drpt: float
        Percentage of neurons that will be used during training process
    save_step: integer
        Periodicity of training result saving (number of iteration)
    log_step: integer
        Periodicity of training result printing (number of iteration)
    batch_size: integer
        Number of images in each mini-batches
    name: object
        String designing the name of the network
    datapath: object
        String designing the relative path to dataset directory
    
    """
    NETWORK_NAME = (name + "_" + weight_policy + "_" + str(nbconv)
                    + "_" + str(nbfullyconn))
    if mode == "train":
        utils.logger.info("Model {} training".format(NETWORK_NAME))
    elif mode == "test":
        utils.logger.info("Model {} testing".format(NETWORK_NAME))
    elif mode == "both":
        utils.logger.info("Model {} training and testing".format(NETWORK_NAME))
    config_file_name = os.path.join("..", "models", NETWORK_NAME + ".json")
    with open(config_file_name) as config_file:
        cnn_hyperparam = json.load(config_file)

    NETWORK_NAME = NETWORK_NAME + "_" + str(image_size)
    n_images = len(os.listdir(os.path.join(datapath, "training",
                                           "input" + "_" + str(image_size))))
    n_val_images = len(os.listdir(os.path.join(datapath, "validation",
                                               "input" + "_"
                                               + str(image_size))))
    n_batches = int(n_images / batch_size)
    n_val_batches = int(n_val_images / batch_size)

    utils.logger.info(("{} images splitted into {} batches (batch_size={})!"
                       "").format(n_images, n_batches, batch_size))

    utils.logger.info("Input size: {}*{}*3".format(image_size, image_size))
    utils.logger.info("Network architecture: ")

    # Data recovering
    train_image_batch, train_label_batch, train_filename_batch = \
    cnn_layers.prepare_data(image_size, 3,
                            batch_size, label_list, datapath,
                            "training", "training_data_pipe")
    val_image_batch, val_label_batch, val_filename_batch = \
    cnn_layers.prepare_data(image_size, 3,
                            batch_size, label_list, datapath,
                            "validation", "valid_data_pipe")

    # Definition of TensorFlow placeholders
    X = tf.placeholder(tf.float32, [None, image_size, image_size,
                                    3], name='X')
    Y = tf.placeholder(tf.float32, [None, len(label_list)], name='Y')
    dropout = tf.placeholder(tf.float32, name='dropout')
    class_w = tf.placeholder(tf.float32, [len(label_list)],
                             name='weights_per_label')

    # Model building
    logits, y_raw_pred, y_pred = cnn_layers.convnet_building(X, cnn_hyperparam,
                                                             image_size,
                                                             3,
                                                             len(label_list),
                                                             dropout,
                                                             NETWORK_NAME,
                                                             nbconv,
                                                             nbfullyconn)
    
    # Loss function design
    output = cnn_layers.define_loss(Y, logits, y_raw_pred, class_w, start_lr,
                                    decay_steps, decay_rate, NETWORK_NAME)

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
            # Prepare the result backup csv file
            db_columns = db.dashboard_columns(label_list)
            utils.make_dir(os.path.join("..", "data", "results", NETWORK_NAME))
            result_file_name = os.path.join(datapath, "results",
                                            NETWORK_NAME, NETWORK_NAME)
            if mode == "both":
                val_result_file_name = os.path.join(datapath, "results",
                                                    NETWORK_NAME,
                                                    NETWORK_NAME + "_val")

            # Train the model
            start_time = time.time()

            if weight_policy == "base":
                w_batch = np.repeat(1.0, len(label_list))
            elif weight_policy == "global":
                label_counter = gr.count_image_per_label(datapath, image_size)
                label_counter = [label_counter[l] for l in label_list]
                w_batch = utils.compute_monotonic_weights(n_images,
                                                          label_counter)
            elif weight_policy == "centeredglobal":
                label_counter = gr.count_image_per_label(datapath, image_size)
                label_counter = [label_counter[l] for l in label_list]
                w_batch = utils.compute_centered_weights(n_images, label_counter)

            if nb_iter is None:
                nb_iter = n_batches * nb_epochs
            for step in range(initial_step, nb_iter):
                X_batch, Y_batch = sess.run([train_image_batch,
                                             train_label_batch])
                if weight_policy == "batch":
                    label_counter = [sum(s) for s in np.transpose(Y_batch)]
                    w_batch = utils.compute_monotonic_weights(batch_size,
                                                              label_counter)
                elif weight_policy == "centeredbatch":
                    label_counter = [sum(s) for s in np.transpose(Y_batch)]
                    w_batch = utils.compute_centered_weights(batch_size,
                                                             label_counter)
                fd = {X: X_batch, Y: Y_batch, dropout: 1.0, class_w: w_batch}

                if (step + 1) % log_step == 0 or step == initial_step:
                    Y_pred, loss, bpmll, mll, lr, lo = sess.run([y_pred,
                                                        output["loss"],
                                                        output["bpmll"],
                                                        output["ml_loss"],
                                                                 output["lrate"],
                                                                     logits],
                                                                feed_dict=fd)
                    db_batch = db.dashboard_building(Y_batch, Y_pred)
                    db_batch.insert(0, bpmll)
                    db_batch.insert(0, loss)
                    db_batch.insert(0, step)
                    df_batch = pd.DataFrame(np.array(db_batch).reshape([1,-1]),
                                            columns=db_columns)

                    csv_mode = "w" if step == 0 else "a"
                    df_batch.to_csv(result_file_name + ".csv", index=False,
                                    mode=csv_mode, header=csv_mode=="w")
                    batch_res = db.dashboard_result(df_batch, step)
                    batch_res.to_csv(result_file_name + "_step"
                                     + str(step) + ".csv")

                    if mode == "both":
                        # Run the model on validation dataset
                        partial_val_dashboard = []
                        for val_step in range(n_val_batches):
                            X_val_batch, Y_val_batch = \
                            sess.run([val_image_batch, val_label_batch])
                            fd_val = {X: X_val_batch, Y: Y_val_batch,
                                      dropout: 1.0, class_w: w_batch}
                            Y_pred_val, loss_batch_val, bpmll_val =\
                    sess.run([y_pred, output["loss"], output["bpmll"]],
                            feed_dict=fd_val)
                            db_val_batch = db.dashboard_building(Y_val_batch,
                                                                 Y_pred_val)
                            db_val_batch.insert(0, bpmll_val)
                            db_val_batch.insert(0, loss_batch_val)
                            db_val_batch.insert(0, step)
                            partial_val_dashboard.append(db_val_batch)

                        curval_dashboard = (pd.DataFrame(partial_val_dashboard)
                                            .apply(lambda x: x.mean(), axis=0))
                        curval_dashboard.columns = db_columns
                        curval_dashboard.set_index("epoch")
                        curval_dashboard.to_csv(result_file_name, index=True,
                                                mode=csv_mode,
                                                header=csv_mode=="w")
                        utils.logger.info(("Step {} (lr={:.5f}): loss={:5.1f},"
                                           " mlloss={:5.1f}"
                                           " cm=[{},{},{},{}] (validation: "
                                           "[{},{},{},{}]), "
                                           "acc={:1.3f} (validation: "
                                           "{:1.3f}), tpr={:1.3f}, "
                                           "tnr={:1.3f}, fpr={:1.3f}, "
                                           "fnr={:1.3f}, ppv={:1.3f}, "
                                           "npv={:1.3f}, f1s={:1.3f}")
                                          .format(step, lr, loss, mll,
                                                  db_batch[4], db_batch[5],
                                                  db_batch[6], db_batch[7],
                                                  db_val_batch[4], db_val_batch[5],
                                                  db_val_batch[6], db_val_batch[7],
                                                  db_batch[8], db_val_batch[8],
                                                  db_batch[9],
                                                  db_batch[10], db_batch[11],
                                                  db_batch[12], db_batch[13],
                                                  db_batch[14], db_batch[15]))
                    else:
                        utils.logger.info(("Step {} (lr={:.5f}): loss={:5.1f},"
                                           " mlloss={:5.1f}"
                                           " cm=[{},{},{},{}], "
                                           "acc={:1.3f}, tpr={:1.3f}, "
                                           "tnr={:1.3f}, fpr={:1.3f}, "
                                           "fnr={:1.3f}, ppv={:1.3f}, "
                                           "npv={:1.3f}, f1s={:1.3f}")
                                          .format(step, lr, loss, mll,
                                                  db_batch[4], db_batch[5],
                                                  db_batch[6], db_batch[7],
                                                  db_batch[8], db_batch[9],
                                                  db_batch[10], db_batch[11],
                                                  db_batch[12], db_batch[13],
                                                  db_batch[14], db_batch[15]))

                # Run the model to do a new training iteration
                fd = {X: X_batch, Y: Y_batch, dropout: drpt, class_w: w_batch}
                sess.run(output["optim"], feed_dict=fd)

                # If all training batches have been scanned, save the model
                if (step + 1) % save_step == 0:
                    utils.logger.info(("Checkpoint {}/checkpoints/{}/epoch-{}"
                                       " creation")
                                      .format(datapath, NETWORK_NAME, step))
                    saver.save(sess, global_step=step,
                               save_path=os.path.join(datapath, 'checkpoints',
                                                      NETWORK_NAME, 'epoch'))

            utils.logger.info("Optimization Finished!")
            utils.logger.info("Total time: {:.2f} seconds".format(time.time() -
                                                                  start_time))

        elif mode == "test":
            utils.logger.info(("Test model after {}"
                               " training steps!").format(initial_step))

            # Run the model on validation dataset
            val_dashboard = []
            for val_step in range(n_val_batches):
                X_val_batch, Y_val_batch = sess.run([val_image_batch,
                                                     val_label_batch])
                fd_val = {X: X_val_batch, Y: Y_val_batch, dropout: 1.0,
                          class_w: np.repeat(1.0, len(label_list))}
                Y_pred_val = sess.run([y_pred], feed_dict=fd_val)
                db_val_batch = db.dashboard_building(Y_val_batch,
                                                     Y_pred_val[0])
                db_val_batch.insert(0, val_step)
                val_dashboard.append(db_val_batch)
                if (step + 1) % log_step == 0 or step == initial_step:
                    utils.logger.info(("Step {}: accuracy = {:1.3f}, precision"
                                       " = {:1.3f}, recall = {:1.3f}")
                                      .format(val_step, db_val_batch[2],
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
    parser.add_argument('-b', '--batch-size', required=False, type=int,
                        nargs='?', default=20,
                        help=("The number of images that must be contained "
                              "into a single batch"))
    parser.add_argument('-c', '--nbconv', required=False, type=int,
                        nargs='?', default=2,
                        help=("The number of convolutional layers "
                              "that must be inserted into the network"))
    parser.add_argument('-d', '--datapath', required=False,
                        default="../data", nargs='?',
                        help="""The relative path towards data directory""")
    parser.add_argument('-dn', '--dataset-name', required=False,
                        default="dataset", nargs='?',
                        help=("The json dataset filename, "
                              "without its extension"))
    parser.add_argument('-do', '--dropout', required=False,
                        default=2.0/3, nargs='?',
                        help=("The percentage of dropped out neurons "
                              "during training"))
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
    parser.add_argument('-ls', '--log-step', nargs="?",
                        default=10, type=int,
                        help=("The log periodicity during training process"))
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
                        default=[0.01, 1000, 0.95], type=float,
                        help=("List of learning rate components (starting LR, "
                              "decay steps and decay rate)"))
    parser.add_argument('-s', '--image-size', nargs="?",
                        default=512, type=int,
                        help=("The desired size of images (width = height)"))
    parser.add_argument('-ss', '--save-step', nargs="?",
                        default=100, type=int,
                        help=("The save periodicity during training process"))
    parser.add_argument('-t', '--training-limit', default=None, type=int,
                        help=("Number of training iteration, "
                              "if not specified the model run during "
                              "nb-epochs * nb-batchs iterations"))
    parser.add_argument('-w', '--weights', default=["base"], nargs='+',
                        help=("The weight policy to apply on label "
                              "contributions to loss: either 'base' "
                              "(default case), 'global', 'batch', "
                              "'centeredglobal', 'centeredbatch'"))
    args = parser.parse_args()

    if args.image_size < 256 or args.image_size > 2048:
        utils.logger.error(("Unsupported image size. Please provide a "
                            "reasonable image size (between 256 and 2048"))
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
        label_list = [i for i in range(nb_labels-1)]
    else:
        label_list = args.label_list
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

    # if args.prepare_data:
    #     utils.mapillary_data_preparation(args.datapath, "training",
    #                                      args.image_size, nb_labels)
    #     utils.mapillary_data_preparation(args.datapath, "validation",
    #                                      args.image_size, nb_labels)
    # else:
    #     for n in args.name:
    #         for w in args.weights:
    #             run(args.nbconv, args.nbfullyconn, args.nb_epochs,
    #                 args.training_limit, args.mode, label_list,
    #                 args.image_size, w, args.learning_rate[0],
    #                 args.learning_rate[1], args.learning_rate[2],
    #                 args.dropout, args.save_step, args.log_step,
    #                 args.batch_size, n, args.datapath)

    dataset_filename = os.path.join(args.datapath, args.dataset_name+'.json')
    d = Dataset(args.image_size, os.path.join(args.datapath, "config.json"))
    if os.path.isfile(dataset_filename):
        d.load(dataset_filename)
    else:
        d.populate(os.path.join(args.datapath, "validation"))
        d.save(dataset_filename)

    utils.logger.info(("{} classes in the dataset glossary, {} being focused "
                       "").format(d.get_nb_class(), len(label_list)))
    utils.logger.info(("{} images in the training"
                       "set").format(d.get_nb_images()))

    cnn = ConvolutionalNeuralNetwork(network_name=args.name,
                                     image_size=args.image_size,
                                     nb_channels=3,
                                     batch_size=args.batch_size,
                                     nb_labels=len(label_list))
    cnn.train(d, label_list,
              nb_epochs=args.nb_epochs, nb_iter=args.training_limit,
              log_step=args.log_step, save_step=args.save_step)

    sys.exit(0)
