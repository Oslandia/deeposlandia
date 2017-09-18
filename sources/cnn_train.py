# Raphael Delhome - september 2017

# Convolutional Neural Network with Tensorflow

# The goal of this script is to train a neural network model in order to read
# street scene images produced by Mapillary
# (https://www.mapillary.com/dataset/vistas)

# Four main task will be of interest: data recovering, network conception,
# optimization design and model training.

# Step 0: module imports

import json
import logging
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow as tf
import sys
import time

import bpmll # Multilabel classification lossi
import cnn_layers
import dashboard_building
import utils

# Step 1: parameter definition

# image dimensions (width, height, number of channels)
IMG_SIZE = (768, 576)
IMAGE_HEIGHT  = IMG_SIZE[1]
IMAGE_WIDTH   = IMG_SIZE[0]
NUM_CHANNELS  = 3 # Colored images (RGB)

# hidden layer depth (number of channel per convolutional and fully connected
# layer), kernel dimension, conv layer stride, max pool layer ksize and stride
# Name of the convolutional neural network
for config_file_name in os.listdir(os.path.join("..", "models")):
    NETWORK_NAME = config_file_name.split('.')[0]
    utils.logger.info("Model {} training".format(NETWORK_NAME))
    # NETWORK_NAME = "cnn_mapil_2_0_2_1_1_0"
    config_file_name = os.path.join("..", "models", NETWORK_NAME + ".json")
    with open(os.path.join("..", "models", config_file_name)) as config_file:
        cnn_hyperparam = json.load(config_file)
    L_C1 = cnn_hyperparam["conv1"]["depth"]
    K_C1 = cnn_hyperparam["conv1"]["kernel_size"]
    STR_C1 = cnn_hyperparam["conv1"]["strides"]
    KS_P1 = cnn_hyperparam["pool1"]["kernel_size"]
    STR_P1 = cnn_hyperparam["pool1"]["strides"]
    if "conv2" in cnn_hyperparam.keys():
        L_C2 = cnn_hyperparam["conv2"]["depth"]
        K_C2 = cnn_hyperparam["conv2"]["kernel_size"]
        STR_C2 = cnn_hyperparam["conv2"]["strides"]
    if "pool2" in cnn_hyperparam.keys():
        KS_P2 = cnn_hyperparam["pool2"]["kernel_size"]
        STR_P2 = cnn_hyperparam["pool2"]["strides"]
    if "conv3" in cnn_hyperparam.keys():
        L_C3 = cnn_hyperparam["conv3"]["depth"]
        K_C3 = cnn_hyperparam["conv3"]["kernel_size"]
        STR_C3 = cnn_hyperparam["conv3"]["strides"]
    if "pool3" in cnn_hyperparam.keys():
        KS_P3 = cnn_hyperparam["pool3"]["kernel_size"]
        STR_P3 = cnn_hyperparam["pool3"]["strides"]
    L_FC1 = cnn_hyperparam["fullconn1"]["depth"]
    if "fullconn2" in cnn_hyperparam.keys():
        L_FC2 = cnn_hyperparam["fullconn2"]["depth"]

    # number of output classes
    N_CLASSES = 66
    # number of images per batch
    BATCH_SIZE = 20
    N_BATCHES = int(18000 / BATCH_SIZE) # TODO
    # number of epochs (one epoch = all images have been used for training)
    N_EPOCHS = 1
    # Starting learning rate (it moves following an exponential decay afterwards)
    START_LR = 0.01
    # dropout, i.e. percentage of nodes that are briefly removed during training
    # process
    DROPOUT = 2/3.0
    # printing frequency during training
    SKIP_STEP = 10

    # Step 2: data recovering

    train_image_batch, train_label_batch, train_filename_batch = \
    cnn_layers.prepare_data(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS,
                                   BATCH_SIZE, "training", "training_data_pipe")
    validation_image_batch, validation_label_batch, validation_filename_batch =\
    cnn_layers.prepare_data(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS,
                                   BATCH_SIZE, "validation", "validation_data_pipe")

    # Step 3: Prepare the checkpoint creation

    utils.make_dir('../checkpoints')
    utils.make_dir('../checkpoints/'+NETWORK_NAME)

    # Step 4: create placeholders

    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH,
                                    NUM_CHANNELS], name='X')
    Y = tf.placeholder(tf.float32, [None, N_CLASSES], name='Y')
    dropout = tf.placeholder(tf.float32, name='dropout')

    # Step 5: model building

    conv1 = cnn_layers.conv_layer(X, NUM_CHANNELS, K_C1, L_C1, STR_C1,
                                  1, NETWORK_NAME)
    pool1 = cnn_layers.maxpool_layer(conv1, KS_P1, STR_P1,
                                     1, NETWORK_NAME)
    last_pool = pool1
    last_layer_dim = L_C1
    layer_coefs = [STR_C1, STR_P1]
    if "conv2" in cnn_hyperparam.keys():
        conv2 = cnn_layers.conv_layer(pool1, L_C1, K_C2, L_C2, STR_C2,
                                      2, NETWORK_NAME)
        layer_coefs.append(STR_C2)
    if "pool2" in cnn_hyperparam.keys():
        pool2 = cnn_layers.maxpool_layer(conv2, KS_P2, STR_P2,
                                         2, NETWORK_NAME)
        last_pool = pool2
        last_layer_dim = L_C2
        layer_coefs.append(STR_P2)
    if "conv3" in cnn_hyperparam.keys():
        conv3 = cnn_layers.conv_layer(pool2, L_C2, K_C3, L_C3, STR_C3,
                                      3, NETWORK_NAME)
        layer_coefs.append(STR_C3)
    if "pool3" in cnn_hyperparam.keys():
        pool3 = cnn_layers.maxpool_layer(conv3, KS_P3, STR_P3, 3, NETWORK_NAME)
        last_pool = pool3
        last_layer_dim = L_C3
        layer_coefs.append(STR_P3)
    hidden_layer_dim = cnn_layers.layer_dim(IMAGE_HEIGHT, IMAGE_WIDTH,
                                                   layer_coefs, last_layer_dim)
    fc1 = cnn_layers.fullconn_layer(last_pool, IMAGE_HEIGHT, IMAGE_WIDTH,
                                    hidden_layer_dim, L_FC1,
                                    dropout, 1, NETWORK_NAME)
    if "fullconn2" in cnn_hyperparam.keys():
        fc2 = cnn_layers.fullconn_layer(fc1, IMAGE_HEIGHT, IMAGE_WIDTH,
                                        L_FC1, L_FC2,
                                        dropout, 2, NETWORK_NAME)

    # Output building

    with tf.variable_scope(NETWORK_NAME + '_sigmoid_linear') as scope:
        # Create weights and biases for the final fully-connected layer
        w = tf.get_variable('weights', [L_FC1, N_CLASSES],
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [N_CLASSES],
                            initializer=tf.random_normal_initializer())
        # Compute logits through a simple linear combination
        logits = tf.add(tf.matmul(fc1, w), b)
        # Compute predicted outputs with sigmoid function
        Y_raw_predict = tf.nn.sigmoid(logits)
        Y_predict = tf.to_int32(Y_raw_predict)

    # Step 6: loss function design

    with tf.name_scope(NETWORK_NAME + '_loss'):
        # Cross-entropy between predicted and real values: we use sigmoid instead
        # of softmax as we are in a multilabel classification problem
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits)
        loss = tf.reduce_mean(entropy, name="loss")
        bpmll_loss = bpmll.bp_mll_loss(Y, Y_raw_predict)

    # Step 7: Define training optimizer

    with tf.name_scope(NETWORK_NAME +  '_train'):
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        # Variable learning rate
        lrate = tf.train.exponential_decay(START_LR, global_step,
                                           decay_steps=1000, decay_rate=0.95,
                                           name='learning_rate')
        # Use Adam optimizer with decaying learning rate to minimize cost.
        optimizer = tf.train.AdamOptimizer(lrate).minimize(loss,
                                                           global_step=global_step)

    # Final step: running the neural network

    with tf.Session() as sess:
        # Initialize the tensorflow variables
        # To visualize using TensorBoard
        # tensorboard --logdir="../graphs/"+NETWORK_NAME --port 6006)
        sess.run(tf.global_variables_initializer())
        # Declare a saver instance and a summary writer to store the trained network
        saver = tf.train.Saver(max_to_keep=1)
        writer = tf.summary.FileWriter('../graphs/'+NETWORK_NAME, sess.graph)
        initial_step = global_step.eval(session=sess)
        # Create folders to store checkpoints
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('../checkpoints/'+NETWORK_NAME+'/checkpoint'))
        # If that checkpoint exists, restore from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # Initialize threads to begin batching operations
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Train the model
        start_time = time.time()
        dashboard = []
        best_accuracy = 0
        for index in range(initial_step, N_BATCHES * N_EPOCHS):
            X_batch, Y_batch = sess.run([train_image_batch, train_label_batch])
            X_val_batch, Y_val_batch = sess.run([validation_image_batch,
                                                 validation_label_batch])
            if index % SKIP_STEP == 0:
                loss_batch, bpmll_l, Y_pred = \
                sess.run([loss, bpmll_loss, Y_predict],
                         feed_dict={X: X_batch, Y: Y_batch, dropout: 1.0})
                dashboard_batch = dashboard_building.dashboard_building(Y_batch, Y_pred)
                dashboard_batch.insert(0, bpmll_l)
                dashboard_batch.insert(0, loss_batch)
                dashboard_batch.insert(0, index)
                dashboard.append(dashboard_batch)
                utils.logger.info("""Step {}: loss = {:5.3f}, accuracy={:1.3f}, precision={:1.3f}, recall={:1.3f}""".format(index, loss_batch, dashboard_batch[4], dashboard_batch[5], dashboard_batch[6]))
            if best_accuracy < dashboard_batch[4]:
                best_accuracy = dashboard_batch[4]
                saver.save(sess, '../checkpoints/'+NETWORK_NAME+'/best', index)
            sess.run(optimizer, feed_dict={X: X_batch, Y: Y_batch, dropout: DROPOUT})
        utils.logger.info("Optimization Finished!")
        utils.logger.info("Total time: {:.2f} seconds".format(time.time() - start_time))

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
        # Stop the threads used during the process
        coord.request_stop()
        coord.join(threads)
