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

import json
import logging
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
    # call the script following format 'python3 cnn_train.py configfile.json'
    if len(sys.argv) != 2:
        utils.logger.error("Usage: python3 cnn_train.py <config_filename.json>")
        sys.exit(-1)
    NETWORK_NAME = sys.argv[1]
    # image dimensions (width, height, number of channels)
    IMG_SIZE = (768, 576)
    IMAGE_HEIGHT  = IMG_SIZE[1]
    IMAGE_WIDTH   = IMG_SIZE[0]
    NUM_CHANNELS  = 3 # Colored images (RGB)

    utils.make_dir('../data/checkpoints')
    utils.make_dir('../data/checkpoints/'+NETWORK_NAME)

    utils.logger.info("Model {} training".format(NETWORK_NAME))
    config_file_name = NETWORK_NAME + ".json"
    with open(os.path.join("..", "models", "to_run", config_file_name)) as config_file:
        cnn_hyperparam = json.load(config_file)

    # number of output classes
    N_CLASSES = 66
    # number of images per batch
    BATCH_SIZE = 20
    N_BATCHES = int(len(os.listdir(os.path.join("..", "data", "training", "images"))) / BATCH_SIZE)
    # number of epochs (one epoch = all images have been used for training)
    N_EPOCHS = 1
    # Learning rate tuning (exponential decay)
    START_LR = 0.01
    DECAY_STEPS = 100
    DECAY_RATE = 0.9
    # dropout, i.e. percentage of nodes that are briefly removed during training
    # process
    DROPOUT = 2/3.0
    # printing frequency during training
    SKIP_STEP = 10

    # Data recovering
    train_image_batch, train_label_batch, train_filename_batch = \
    cnn_layers.prepare_data(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS,
                                   BATCH_SIZE, "training", "training_data_pipe")

    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH,
                                    NUM_CHANNELS], name='X')
    Y = tf.placeholder(tf.float32, [None, N_CLASSES], name='Y')
    dropout = tf.placeholder(tf.float32, name='dropout')
    class_w = tf.placeholder(tf.float32, [N_CLASSES],
                             name='weights_per_label')

    # Model building
    last_fc, last_fc_layer_dim = cnn_layers.convnet_building(X, cnn_hyperparam,
                                                             IMG_SIZE[0],
                                                             IMG_SIZE[1],
                                                             NUM_CHANNELS,
                                                             dropout,
                                                             NETWORK_NAME, 6, 2)
    
    # Output building
    with tf.variable_scope(NETWORK_NAME + '_sigmoid_linear') as scope:
        # Create weights and biases for the final fully-connected layer
        w = tf.get_variable('weights', [last_fc_layer_dim, N_CLASSES],
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [N_CLASSES],
                            initializer=tf.random_normal_initializer())
        # Compute logits through a simple linear combination
        logits = tf.add(tf.matmul(last_fc, w), b)
        # Compute predicted outputs with sigmoid function
        Y_raw_predict = tf.nn.sigmoid(logits)
        Y_predict = tf.to_int32(tf.round(Y_raw_predict))

    # Loss function design
    with tf.name_scope(NETWORK_NAME + '_loss'):
        # entropy = tf.nn.weighted_cross_entropy_with_logits(targets=Y,
        #                                                    logits=logits,
        #                                                    pos_weight=1.5)
        # Tensorflow definition of sigmoid cross-entropy:
        # (tf.maximum(logits, 0) - logits*Y + tf.log(1+tf.exp(-tf.abs(logits))))
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,
                                                          logits=logits)
        weighted_entropy = tf.multiply(class_w, entropy)
        loss = tf.reduce_mean(entropy, name="loss")
        bpmll_loss = bpmll.bp_mll_loss(Y, Y_raw_predict)

    # Define training optimizer
    with tf.name_scope(NETWORK_NAME +  '_train'):
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        # Variable learning rate
        lrate = tf.train.exponential_decay(START_LR, global_step,
                                           decay_steps=DECAY_STEPS,
                                           decay_rate=DECAY_RATE,
                                           name='learning_rate')
        # Use Adam optimizer with decaying learning rate to minimize cost.
        optimizer = tf.train.AdamOptimizer(lrate).minimize(loss,
                                                           global_step=global_step)

    # Running the neural network
    with tf.Session() as sess:
        # Initialize the tensorflow variables
        # To visualize using TensorBoard
        # tensorboard --logdir="../graphs/"+NETWORK_NAME --port 6006)
        sess.run(tf.global_variables_initializer())
        # Declare a saver instance and a summary writer to store the network
        saver = tf.train.Saver(max_to_keep=1)
        writer = tf.summary.FileWriter('../graphs/'+NETWORK_NAME, sess.graph)
        
        # Create folders to store checkpoints
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('../data/checkpoints/' + NETWORK_NAME + '/checkpoint'))
        # If that checkpoint exists, restore from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            utils.logger.info("Recover model state from {}".format(ckpt.model_checkpoint_path))
            utils.logger.info("Global step = {}".format(global_step.eval(session=sess)))
        initial_step = global_step.eval(session=sess)

        # Initialize threads to begin batching operations
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Train the model
        start_time = time.time()
        dashboard = []
        best_accuracy = 0
        for index in range(initial_step, N_BATCHES * N_EPOCHS):
            X_batch, Y_batch = sess.run([train_image_batch, train_label_batch])
            # Case 1: unweighted loss
            # w_batch = tf.ones_like(Y_batch).eval()
            # Case 2: globally weighted loss
            # label_counter = glossary_reading.NB_IMAGE_PER_LABEL
            # w_batch = [min(math.log(0.5 * BATCH_SIZE * N_BATCHES / l), 10.0)
            #            for l in label_counter]
            # Case 3: batch weighted loss
            label_counter = [sum(s) for s in np.transpose(Y_batch)]
            w_batch = [min(math.log(0.5 * BATCH_SIZE / l), 100.0)
                       for l in label_counter]
            # Case 4: centered globally weighted loss
            # label_counter = glossary_reading.NB_IMAGE_PER_LABEL 
            # w_batch = [(math.log(1 + 0.5 * (l - (BATCH_SIZE * N_BATCHES) / 2)**2) / (BATCH_SIZE * N_BATCHES)) for l in label_counter]
            # Case 5: centered batch weighted loss
            # label_counter = [sum(s) for s in np.transpose(Y_batch)]
            # w_batch = [math.log(1 + 0.5 * (l - BATCH_SIZE/2)**2 / BATCH_SIZE)
            #            for l in label_counter]

            if (index + 1) % SKIP_STEP == 0 or index == initial_step:
                Y_pred, loss_batch, bpmll_l, lr = sess.run([Y_predict, loss,
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

                utils.logger.info("""Step {} (lr={:1.3f}): loss = {:5.3f}, accuracy={:1.3f}, precision={:1.3f}, recall={:1.3f}""".format(index, lr, loss_batch, dashboard_batch[4], dashboard_batch[5], dashboard_batch[6]))

            sess.run(optimizer,
                     feed_dict={X: X_batch, Y: Y_batch, dropout: DROPOUT,
                                class_w: w_batch})
            
            if (index + 1) % N_BATCHES == 0:
                utils.logger.info("Checkpoint ../data/checkpoints/{}/epoch-{} creation".format(NETWORK_NAME, index))
                saver.save(sess, '../data/checkpoints/'+NETWORK_NAME+'/epoch',
                           index)
                
        utils.logger.info("Optimization Finished!")
        utils.logger.info("Total time: {:.2f} seconds".format(time.time() - start_time))

        if initial_step < N_BATCHES * N_EPOCHS:
            # The results are stored as a pandas dataframe and saved on the file
            # system
            dashboard_columns = ["epoch", "loss", "bpmll_loss", "hamming_loss",
                                 "accuracy", "precision", "recall", "F_measure"]
            dashboard_columns_by_label = [["accuracy_label"+str(i),
                                           "precision_label"+str(i),
                                           "recall_label"+str(i)]
                                          for i in range(len(Y_batch[0]))]
            dashboard_columns_by_label = utils.unnest(dashboard_columns_by_label)
            dashboard_columns = dashboard_columns + dashboard_columns_by_label
            param_history = pd.DataFrame(dashboard, columns = dashboard_columns)
            param_history = param_history.set_index("epoch")
            utils.make_dir(os.path.join("..", "data", "results"))
            result_file_name = os.path.join("..", "data", "results", NETWORK_NAME + ".csv")
            if initial_step == 0:
                param_history.to_csv(result_file_name, index=True)
            else:
                param_history.to_csv(result_file_name,
                                     index=True,
                                     mode='a',
                                     header=False)

            # Training results are then saved as a multiplot
            complete_dashboard = pd.read_csv(result_file_name)
            plot_file_name = os.path.join("..", "images", NETWORK_NAME + "_s" + str(global_step.eval(session=sess)) + ".png")
            dashboard_building.plot_dashboard(complete_dashboard,
                                              plot_file_name)
            
        # Stop the threads used during the process
        coord.request_stop()
        coord.join(threads)
