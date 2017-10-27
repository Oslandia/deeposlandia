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
 *   License along with this library; if not, see <http://www.gnu.org/licenses/>.
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

import bpmll # Multilabel classification loss
import cnn_layers
import dashboard_building
import glossary_reading
import utils

if __name__ == '__main__':
    # Manage argument parsing
    parser = argparse.ArgumentParser(description="Convolutional Neural Network on street-scene images")
    parser.add_argument('-c', '--nbconv', required=True, nargs='?', type=int,
                        help="""The number of convolutional layers that must be
    inserted into the network""")
    parser.add_argument('-d', '--datapath', required=False,
                        default="../data", nargs='?',
                        help="""The relative path towards data directory""")
    parser.add_argument('-e', '--nb-epochs', required=False, type=int,
                        default=5, nargs='?',
                        help="""The number of training epochs (one epoch means
                        scanning each training image once)""")
    parser.add_argument('-f', '--nbfullyconn', required=True, type=int,
                        nargs='?',
                        help="""The number of fully-connected layers that must
    be inserted into the network""")
    parser.add_argument('-m', '--mode', required=False,
                        default="train", nargs='?',
                        help="""The network running mode ('train', 'test', 'both'""")
    parser.add_argument('-n', '--name', required=False,
                        default="cnn_mapil", nargs='?',
                        help="""The model name that will be used for results,
                        checkout and graph storage on file system""")
    parser.add_argument('-w', '--weights', required=False,
                        default="base", nargs='?',
                        help="""The weight policy to apply on label
                        contributions to loss: either 'base' (default case),
                        'global', 'batch', 'centered_global', 'centered_batch'""")
    args = parser.parse_args()

    if args.mode not in ["train", "test", "both"]:
       utils.logger.error("""Unsupported running mode. Please choose amongst 'train', 'test' or 'both'.""")
       sys.exit(1)
    if args.weights not in ["base", "global", "batch", "centered_batch", "centered_global"]:
       utils.logger.error("""Unsupported weighting policy. Please choose amongst 'basis', 'global', 'batch', 'centered_global' or 'centered_batch'.""")
       utils.logger.info("'base': Regular weighting scheme...")
       utils.logger.info("""'global': Label contributions to loss are weighted with respect to label popularity within the dataset (decreasing weights)...""")
       utils.logger.info("""'batch': Label contributions to loss are weighted with respect to label popularity within the dataset (convex weights with min at 50%)...""")
       utils.logger.info("""'centeredbatch': Label contributions to loss are weighted with respect to label popularity within each batch (decreasing weights)...""")
       utils.logger.info("""'centeredglobal': Label contributions to loss are weighted with respect to label popularity within each batch (convex weights with min at 50%)...""")
       sys.exit(1)

    NETWORK_NAME = (args.name + "_" + str(args.nbconv) + "_0_"
                    + str(args.nbconv) + "_0_"
                    + str(args.nbfullyconn) + "_0")
    if args.mode == "train":
        utils.logger.info("Model {} training".format(NETWORK_NAME))
    elif args.mode == "test":
        utils.logger.info("Model {} testing".format(NETWORK_NAME))
    elif args.mode == "both":
        utils.logger.info("Model {} training and testing".format(NETWORK_NAME))

    config_file_name = NETWORK_NAME + ".json"
    with open(os.path.join("..", "models", config_file_name)) as config_file:
        cnn_hyperparam = json.load(config_file)

    # image dimensions (width, height, number of channels)
    IMG_SIZE = (768, 576)
    IMAGE_HEIGHT  = IMG_SIZE[1]
    IMAGE_WIDTH   = IMG_SIZE[0]
    NUM_CHANNELS  = 3 # Colored images (RGB)

    # number of output classes
    N_CLASSES = glossary_reading.LABELS.shape[1]
    # number of images per batch
    BATCH_SIZE = 20
    N_BATCHES = int(len(os.listdir(os.path.join(args.datapath, "training",
                                                "images")))
                    / BATCH_SIZE)
    N_VAL_BATCHES = int(len(os.listdir(os.path.join(args.datapath,
                                                    "validation", "images")))
                        / BATCH_SIZE)
    # learning rate tuning (exponential decay)
    START_LR = 0.01
    DECAY_STEPS = 100
    DECAY_RATE = 0.9
    # percentage of nodes that are briefly removed during training process
    DROPOUT = 2/3.0
    # number of epochs (one epoch = all images have been used for training)
    N_EPOCHS = args.nb_epochs
    # printing frequency during training
    SKIP_STEP = 10

    # Data recovering
    train_image_batch, train_label_batch, train_filename_batch = \
    cnn_layers.prepare_data(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS,
                                   BATCH_SIZE, "training", "training_data_pipe")
    valid_image_batch, valid_label_batch, valid_filename_batch = \
    cnn_layers.prepare_data(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS,
                                   BATCH_SIZE, "validation", "valid_data_pipe")

    # Definition of TensorFlow placeholder
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH,
                                    NUM_CHANNELS], name='X')
    Y = tf.placeholder(tf.float32, [None, N_CLASSES], name='Y')
    dropout = tf.placeholder(tf.float32, name='dropout')
    class_w = tf.placeholder(tf.float32, [N_CLASSES],
                             name='weights_per_label')

    # Model building
    logits, y_raw_pred, y_pred = cnn_layers.convnet_building(X, cnn_hyperparam,
                                                             IMG_SIZE[0],
                                                             IMG_SIZE[1],
                                                             NUM_CHANNELS,
                                                             N_CLASSES,
                                                             dropout,
                                                             NETWORK_NAME,
                                                             args.nbconv,
                                                             args.nbfullyconn)

    # Loss function design
    with tf.name_scope(NETWORK_NAME + '_loss'):
        # Tensorflow definition of sigmoid cross-entropy:
        # (tf.maximum(logits, 0) - logits*Y + tf.log(1+tf.exp(-tf.abs(logits))))
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,
                                                          logits=logits)
        weighted_entropy = tf.multiply(class_w, entropy)
        loss = tf.reduce_mean(weighted_entropy, name="loss")
        bpmll_loss = bpmll.bp_mll_loss(Y, y_raw_pred)
        # Alternative way of measuring a weighted cross-entropy (weighting true
        # and false labels, but not label contributions to loss):
        # entropy = tf.nn.weighted_cross_entropy_with_logits(targets=Y,
        #                                                    logits=logits,
        #                                                    pos_weight=1.5)
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                                  name='global_step')
        lrate = tf.train.exponential_decay(START_LR, global_step,
                                           decay_steps=DECAY_STEPS,
                                           decay_rate=DECAY_RATE,
                                           name='learning_rate')
        optimizer = tf.train.AdamOptimizer(lrate).minimize(loss, global_step)

    # Running the neural network
    with tf.Session() as sess:
        # Initialize the tensorflow variables
        sess.run(tf.global_variables_initializer())
        # Declare a saver instance and a summary writer to store the network
        saver = tf.train.Saver(max_to_keep=1)
        graph_path = os.path.join(args.datapath, 'graphs', NETWORK_NAME)
        writer = tf.summary.FileWriter(graph_path, sess.graph)
        
        # Create folders to store checkpoints
        ckpt_path = os.path.join(args.datapath, 'checkpoints', NETWORK_NAME)
        utils.make_dir(os.path.dirname(ckpt_path))
        utils.make_dir(ckpt_path)
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        # If that checkpoint exists, restore from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            utils.logger.info("Recover model state from {}".format(ckpt.model_checkpoint_path))
        initial_step = global_step.eval(session=sess)

        # Initialize threads to begin batching operations
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        if args.mode in ["train", "both"]:
            # Train the model
            start_time = time.time()
            dashboard = []
            val_dashboard = []
            if args.weights == "base":
                w_batch = np.repeat(1.0, N_CLASSES)
            elif args.weights == "global":
                label_counter = glossary_reading.NB_IMAGE_PER_LABEL
                w_batch = [min(math.log(0.5 * BATCH_SIZE * N_BATCHES / l), 10.0)
                           for l in label_counter]
            elif args.weights == "centered_global":
                label_counter = glossary_reading.NB_IMAGE_PER_LABEL
                w_batch = [(math.log(1 + 0.5 * (l - (BATCH_SIZE * N_BATCHES) / 2)**2) / (BATCH_SIZE * N_BATCHES)) for l in label_counter]
            for index in range(initial_step, N_BATCHES * N_EPOCHS):
                X_batch, Y_batch = sess.run([train_image_batch, train_label_batch])
                if args.weights == "batch":
                    label_counter = [sum(s) for s in np.transpose(Y_batch)]
                    w_batch = [min(math.log(0.5 * BATCH_SIZE / l), 100.0)
                               for l in label_counter]
                elif args.weights == "centered_batch":
                    label_counter = [sum(s) for s in np.transpose(Y_batch)]
                    w_batch = [math.log(1 + 0.5 * (l - BATCH_SIZE/2)**2 / BATCH_SIZE)
                               for l in label_counter]
                    
                if (index + 1) % SKIP_STEP == 0 or index == initial_step:
                    Y_pred, loss_batch, bpmll_l, lr = sess.run([y_pred, loss,
                                                                bpmll_loss, lrate],
                                                       feed_dict={X: X_batch,
                                                                  Y: Y_batch,
                                                                  dropout: 1.0,
                                                                  class_w: w_batch})
                    dashboard_batch = dashboard_building.dashboard_building(Y_batch, Y_pred)
                    dashboard_batch.insert(0, bpmll_l)
                    dashboard_batch.insert(0, loss_batch)
                    dashboard_batch.insert(0, index)
                    dashboard.append(dashboard_batch)

                    if args.mode == "both":
                        # Run the model on validation dataset
                        partial_val_dashboard = []
                        for val_index in range(N_VAL_BATCHES):
                            X_val_batch, Y_val_batch = sess.run([valid_image_batch,
                                                                 valid_label_batch])
                            Y_pred_val, loss_batch_val, bpmll_val = sess.run([y_pred, loss, bpmll_loss], feed_dict={X: X_val_batch, Y: Y_val_batch, dropout: 1.0, class_w: w_batch})
                            db_val_batch = dashboard_building.dashboard_building(Y_val_batch, Y_pred_val)
                            db_val_batch.insert(0, bpmll_val)
                            db_val_batch.insert(0, loss_batch_val)
                            db_val_batch.insert(0, index)
                            partial_val_dashboard.append(db_val_batch)
                        val_cur_dashboard = list(pd.DataFrame(partial_val_dashboard)
                                                 .apply(lambda x: x.mean(), axis=0))
                        val_dashboard.append(val_cur_dashboard)
                        utils.logger.info("""Step {} (lr={:1.3f}): loss = {:5.3f}, accuracy={:1.3f} (validation: {:5.3f}), precision={:1.3f}, recall={:1.3f}""".format(index, lr, loss_batch, dashboard_batch[4], val_cur_dashboard[4], dashboard_batch[5], dashboard_batch[6]))
                    else:
                        utils.logger.info("""Step {} (lr={:1.3f}): loss = {:5.3f}, accuracy={:1.3f}, precision={:1.3f}, recall={:1.3f}""".format(index, lr, loss_batch, dashboard_batch[4], dashboard_batch[5], dashboard_batch[6]))

                # Run the model to do a new training iteration
                sess.run(optimizer, feed_dict={X: X_batch, Y: Y_batch,
                                               dropout: DROPOUT,
                                               class_w: w_batch})

                # If all training batches have been scanned, save the training state
                if (index + 1) % N_BATCHES == 0:
                    utils.logger.info("Checkpoint {}/checkpoints/{}/epoch-{} creation".format(args.datapath, NETWORK_NAME, index))
                    saver.save(sess, global_step=index,
                               save_path=os.path.join(args.datapath,
                                                      'checkpoints',
                                                      NETWORK_NAME, 'epoch'))

            utils.logger.info("Optimization Finished!")
            utils.logger.info("Total time: {:.2f} seconds".format(time.time() - start_time))

            if initial_step < N_BATCHES * N_EPOCHS:
                # The results are stored as a df and saved on the file system
                dashboard_columns = ["epoch", "loss", "bpmll_loss",
                                     "hamming_loss", "accuracy", "precision",
                                     "recall", "F_measure"]
                dashboard_columns_by_label = [["accuracy_label"+str(i),
                                               "precision_label"+str(i),
                                               "recall_label"+str(i)]
                                              for i in range(N_CLASSES)]
                dashboard_columns_by_label = utils.unnest(dashboard_columns_by_label)
                db_columns = dashboard_columns + dashboard_columns_by_label
                param_history = pd.DataFrame(dashboard, columns=db_columns)
                param_history = param_history.set_index("epoch")
                val_param_history = pd.DataFrame(val_dashboard, columns=db_columns)
                val_param_history = val_param_history.set_index("epoch")
                utils.make_dir(os.path.join("..", "data", "results"))
                result_file_name = os.path.join("..", "data", "results", NETWORK_NAME + ".csv")
                val_result_file_name = os.path.join("..", "data", "results", NETWORK_NAME + "_validation.csv")
                if initial_step == 0:
                    param_history.to_csv(result_file_name, index=True)
                    if args.mode in ["both", "test"]:
                        val_param_history.to_csv(val_result_file_name, index=True)
                else:
                    param_history.to_csv(result_file_name,
                                         index=True,
                                         mode='a',
                                         header=False)
                    if args.mode in ["both", "test"]:
                        val_param_history.to_csv(val_result_file_name,
                                                 index=True,
                                                 mode='a',
                                                 header=False)

                # Training results are then saved as a multiplot
                complete_dashboard = pd.read_csv(result_file_name)
                plot_file_name = os.path.join("..", "images",
                                              (NETWORK_NAME + "_s"
                                               + str(global_step.eval(session=sess)) + ".png"))
                dashboard_building.plot_dashboard(complete_dashboard,
                                                  plot_file_name)
            
        # Stop the threads used during the process
        coord.request_stop()
        coord.join(threads)

    sys.exit(0)
