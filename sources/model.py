#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
/**
 *   Raphael Delhome - december 2017
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

import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops

import dataset
import utils

class ConvolutionalNeuralNetwork(object):

    def __init__(self, network_name="mapillary", image_size=512, nb_channels=3,
                 batch_size=128, nb_labels=65, learning_rate=1e-3):
        self._network_name = network_name
        self._image_size = image_size
        self._nb_channels = nb_channels
        self._nb_labels = nb_labels
        self._batch_size = batch_size
        self._learning_rate = learning_rate

    def get_network_name(self):
        return self._network_name

    def get_image_size(self):
        return self._image_size

    def get_nb_channels(self):
        return self._nb_channels

    def get_batch_size(self):
        return self._batch_size
    
    def get_learning_rate(self):
        return self._learning_rate
    
    def get_nb_labels(self):
        return self._nb_labels

    def create_weights(self, shape):
        return tf.get_variable('weights', shape,
                               initializer=tf.truncated_normal_initializer())

    def create_biases(self, shape):
        return tf.get_variable('biases', shape,
                               initializer=tf.constant_initializer(0.0))

    def convolutional_layer(self, counter, input_layer,
                            input_layer_depth, kernel_dim,
                            layer_depth, strides=[1, 1, 1, 1], padding='SAME'):
        """Build a convolutional layer as a Tensorflow object,
        for a convolutional neural network

        Parameters
        ----------
        input_layer: tensor
            input tensor, i.e. the placeholder that represents input data if
        the convolutional layer is the first of the network, the previous layer
        output otherwise
        input_layer_depth: integer
            input tensor channel number (3 for RGB images, more if another
        convolutional layer precedes the current one)
        kernel_dim: integer
            Dimension of the convolution kernel (only the first one, the kernel
        being considered as squared; and its last dimensions being given by
        previous and current layer depths)
        layer_depth: integer
            current layer channel number
        conv_strides: list
            Dimensions of the convolution stride operation defined as [1,a,a,1]
        where a is the shift (in pixels) between each convolution operation
        counter: integer
            Convolutional layer counter (for scope name unicity)
        """
        with tf.variable_scope(self._network_name+'_conv'+str(counter)) as scope:
            w = self.create_weights([kernel_dim, kernel_dim,
                                     input_layer_depth, layer_depth])
            b = self.create_biases([layer_depth])
            conv = tf.nn.conv2d(input_layer, w, strides=strides,
                                padding=padding)
            return tf.nn.relu(tf.add(conv, b), name=scope.name)

    
    def maxpooling_layer(self, counter, input_layer, kernel_dim,
                         stride=2, padding='SAME'):
        """Build a max pooling layer as a Tensorflow object,
        into the convolutional neural network

        Parameters
        ----------
        counter: integer
            Max pooling layer counter (for scope name unicity)
        input_layer: tensor
            Pooling layer input; output of the previous layer into the network
        kernel_dim: list
            Dimension of the pooling kernel, defined as [1, a, a, 1]
        where a is the main kernel dimension (in pixels)
        stride: list
            Dimension of the pooling stride, defined as [1, a, a, 1]
        where a is the shift between each pooling operation

        """
        with tf.variable_scope(self._network_name + '_pool' + str(counter)) as scope:
            return tf.nn.max_pool(input_layer,
                                  ksize=[1, kernel_dim, kernel_dim, 1],
                                  strides=[1,stride,stride,1], padding=padding)

    def get_last_conv_layer_dim(self, strides, last_layer_depth):
        """Consider the current layer depth as the function of previous layer
        hyperparameters, so as to reshape it as a single dimension layer

        Parameters
        ----------
        strides: list
            list of previous layer hyperparameters, that have an impact on the
        current layer depth
        last_layer_depth: integer
            depth of the last layer in the network

        """
        return last_layer_depth * (int(self._image_size/strides) ** 2)

    def fullyconnected_layer(self, counter, input_layer,
                             last_layer_dim, layer_depth, t_dropout=1.0):
        """Build a fully-connected layer as a tensor, into the convolutional
                       neural network

        Parameters
        ----------
        input_layer: tensor
            Fully-connected layer input; output of the previous layer into the network
        last_layer_dim: integer
            previous layer depth, into the network
        layer_depth: integer
            full-connected layer depth
        t_dropout: tensor
            tensor corresponding to the neuron keeping probability during dropout operation
        counter: integer
            fully-connected layer counter (for scope name unicity)

        """
        with tf.variable_scope(self._network_name + '_fc' + str(counter)) as scope:
            reshaped = tf.reshape(input_layer, [-1, last_layer_dim])
            w = self.create_weights([last_layer_dim, layer_depth])
            b = self.create_biases([layer_depth])
            return tf.nn.relu(tf.add(tf.matmul(reshaped, w), b), name='relu')
            # return tf.nn.dropout(fc, t_dropout, name='relu_with_dropout')

    def output_layer(self, input_layer, input_layer_dim):
        """Build an output layer to a neural network with a sigmoid final
        activation function (softmax if there is only one label to predict);
        return final network scores (logits) as well as predictions

        Parameters
        ----------
        input_layer: tensor
            Previous layer within the neural network (last hidden layer)
        input_layer_dim: integer
            Dimension of the previous neural network layer

        """
        with tf.variable_scope(self._network_name + '_output_layer') as scope:
            w = self.create_weights([input_layer_dim, self._nb_labels])
            b = self.create_biases([self._nb_labels])
            logits = tf.add(tf.matmul(input_layer, w), b, name="logits")
            Y_raw_predict = tf.nn.sigmoid(logits, name="y_pred_raw")
            return {"logits": logits, "y_pred": Y_raw_predict}

    def add_layers(self, X):
        """Build the structure of a convolutional neural network from image data X
        to the last hidden layer, this layer being returned by this method

        Parameters
        ----------
        X: tensorflow.placeholder
            Image data with a shape [batch_size, width, height, nb_channels]
        nb_labels: integer
            number of output classes (labels)
        """

        layer = self.convolutional_layer(1, X, self._nb_channels, 8, 16)
        layer = self.maxpooling_layer(1, layer, 2, 2)
        layer = self.convolutional_layer(2, layer, 16, 8, 16)
        layer = self.maxpooling_layer(2, layer, 2, 2)
        layer = self.convolutional_layer(3, layer, 16, 8, 32)
        layer = self.maxpooling_layer(3, layer, 2, 2)
        layer = self.convolutional_layer(4, layer, 32, 8, 32)
        layer = self.maxpooling_layer(4, layer, 2, 2)
        layer = self.convolutional_layer(5, layer, 32, 8, 64)
        layer = self.maxpooling_layer(5, layer, 2, 2)
        layer = self.convolutional_layer(6, layer, 64, 8, 64)
        layer = self.maxpooling_layer(6, layer, 2, 2)
        last_layer_dim = self.get_last_conv_layer_dim(64, 64)
        layer = self.fullyconnected_layer(1, layer, last_layer_dim, 1024, 0.75)
        layer = self.fullyconnected_layer(2, layer, 1024, 1024, 0.75)
        return self.output_layer(layer, 1024)

    def compute_loss(self, y_true, logits, y_raw_p):
        """Define the loss tensor as well as the optimizer; it uses a decaying
        learning rate following the equation

        Parameters
        ----------
        y_true: tensor
            True labels (1 if the i-th label is true for j-th image, 0 otherwise)
        logits: tensor
            Logits computed by the model (scores associated to each labels for a
        given image)
        y_raw_p: tensor
            Raw values computed for outputs (float), before transformation into 0-1
        """

        with tf.name_scope(self._network_name + '_loss'):
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                              logits=logits)
            return tf.reduce_mean(entropy, name="loss")

    def optimize(self, loss):
        """Define the loss tensor as well as the optimizer; it uses a decaying
        learning rate following the equation

        Parameters
        ----------
        loss: tensor
        Tensor that represents the neural network loss function
        """
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                                  name='global_step')
        opt = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        optimizer = opt.minimize(loss, global_step)
        return {"gs": global_step, "lrate": lrate, "optim": optimizer}

    def build(self):
        """ Build the convolutional neural network structure from input
        placeholders to loss function optimization

        """
        X = tf.placeholder(tf.float32, name='X',
                           shape=[None, self._image_size,
                                  self._image_size, self._nb_channels])
        Y = tf.placeholder(tf.float32, name='Y', shape=[None, len(label_list)])
        dropout = tf.placeholder(tf.float32, name='dropout')

        output = self.add_layer(X, len(label_list))

        loss = self.compute_loss(Y, output["logits"], output["y_pred"])
        return self.optimize(loss)

    def define_batch(self, labels_of_interest, datapath, dataset_type):
        """Insert images and labels in Tensorflow batches

        Parameters
        ----------
        labels_of_interest: list
            List of label indices on which a model will be trained
        datapath: object
            String designing the relative path to data
        dataset_type: object
            string designing the considered dataset
        (`training`, `validation` or `testing`)

        """
        INPUT_PATH = os.path.join(datapath, dataset_type,
                                  "input_" + str(self._image_size))
        OUTPUT_PATH = os.path.join(datapath, dataset_type,
                                   "output_" + str(self._image_size))
        with tf.variable_scope(self._network_name+"_"+dataset_type+"_pipe") as scope:
            filepaths = os.listdir(INPUT_PATH)
            filepaths.sort()
            filepaths = [os.path.join(INPUT_PATH, fp) for fp in filepaths]
            images = ops.convert_to_tensor(filepaths, dtype=tf.string,
                                           name=dataset_type+"_images")
            # Reading labels
            df_labels = pd.read_csv(os.path.join(OUTPUT_PATH, "labels.csv"))
            labels = utils.extract_features(df_labels, "label").values
            labels = labels[:, labels_of_interest]
            labels = ops.convert_to_tensor(labels, dtype=tf.int16,
                                           name=dataset_type+"_labels")
            # Create input queues
            input_queue = tf.train.slice_input_producer([images, labels],
                                                        shuffle=False)
            # Process path and string tensor into an image and a label
            file_content = tf.read_file(input_queue[0])
            image = tf.image.decode_jpeg(file_content,
                                         channels=self._nb_channels)
            image.set_shape([self._image_size,
                             self._image_size,
                             self._nb_channels])
            image = tf.div(image, 255) # Data normalization
            label = input_queue[1]
            # Collect batches of images before processing
            return tf.train.batch([image, label],
                                  batch_size=self._batch_size,
                                  num_threads=4)

    def train(self, dataset, nb_epochs):
        """
        """
        train_image_batch, train_label_batch = self.define_batch(label_list,
                                                                 datapath,
                                                                 "training")
        output = self.build()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            initial_step = output["gs"].eval(session=sess)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for step in range(initial_step, nb_iter):
                X_batch, Y_batch = sess.run([train_image_batch,
                                             train_label_batch])
                fd = {X: X_batch, Y: Y_batch, dropout: drpt, class_w: w_batch}
                sess.run(output["optim"], feed_dict=fd)

            coord.request_stop()
            coord.join(threads)
