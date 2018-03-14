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

import abc
import json
import math
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
import time

import dataset
import utils

class ConvolutionalNeuralNetwork(metaclass=abc.ABCMeta):

    def __init__(self, network_name="mapillary", image_size=512, nb_channels=3,
                 nb_labels=65, learning_rate=[1e-3], monitoring_level=1):
        """ Class constructor
        """
        self._network_name = network_name
        self._image_size = image_size
        self._nb_channels = nb_channels
        self._nb_labels = nb_labels
        self._value_ops = {}
        self._update_ops = {}
        self._X = tf.placeholder(tf.float32, name='X',
                                 shape=[None, self._image_size,
                                        self._image_size, self._nb_channels])
        self._dropout = tf.placeholder(tf.float32, name="dropout")
        self._is_training = tf.placeholder(tf.bool, name="is_training")
        self._batch_size = tf.placeholder(tf.int32, name="batch_size")
        self._global_step = tf.Variable(0, dtype=tf.int32,
                                        trainable=False, name='global_step')
        if len(learning_rate) == 1:
            self._learning_rate = learning_rate[0]
        else:
            self._learning_rate = tf.train.exponential_decay(learning_rate[0],
                                                             self._global_step,
                                                             decay_steps=learning_rate[1],
                                                             decay_rate=learning_rate[2],
                                                             name='learning_rate')

        self._monitoring = monitoring_level
        self._chrono = {"monitoring": {},
                        "validation": {},
                        "backprop": {},
                        "backup": {}}

    def get_network_name(self):
        """ `_network_name` getter
        """
        return self._network_name

    def get_image_size(self):
        """ `_image_size` getter
        """
        return self._image_size

    def get_nb_channels(self):
        """ `_nb_channels` getter
        """
        return self._nb_channels
    
    def get_learning_rate(self):
        """ `_learning_rate` getter
        """
        return self._learning_rate
    
    def get_nb_labels(self):
        """ `_nb_labels` getter
        """
        return self._nb_labels

    def create_weights(self, shape):
        """ Create weight variables of dimension `shape`, and initialize them
        with a random truncated normal draw; this function is typically called
        when creating neural network layers (convolutional, fully-connected...)

        Parameter:
        ----------
        shape: list
            List of integers describing the weight shapes (ex: [2], [3, 5]...)
        """
        w = tf.Variable(tf.truncated_normal(shape,
                                            stddev=1.0/math.sqrt(shape[0])),
                        name="weights",
                        trainable=True)
        if self._monitoring >= 3:
            tf.summary.histogram("weights", w)
        return w

    def create_biases(self, shape):
        """ Create biases variables of dimension `shape`, and initialize them
        as zero-constant; this function is typically called when creating
        neural network layers (convolutional, fully-connected...)

        Parameter:
        ----------
        shape: list
            List of integers describing the biases shapes (ex: [2], [3, 5]...)
        """
        b = tf.Variable(tf.zeros(shape), name="biases", trainable=True)
        if self._monitoring >= 3:
            tf.summary.histogram("biases", b)
        return b

    def convolutional_layer(self, counter, is_training, input_layer,
                            input_layer_depth, kernel_dim,
                            layer_depth, strides=[1, 1, 1, 1], padding='SAME'):
        """Build a convolutional layer as a Tensorflow object,
        for a convolutional neural network

        Parameters
        ----------
        counter: integer
            Convolutional layer counter (for scope name unicity)
        is_training: tensor
            Boolean tensor that indicates the phase (training, or not); if training, batch
        normalization on batch statistics, otherwise batch normalization on population statistics
        (see `tf.layers.batch_normalization()` doc)
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
        strides: list
            Dimensions of the convolution stride operation defined as [1,a,a,1]
        where a is the shift (in pixels) between each convolution operation
        padding: object
            String designing the padding mode ('SAME', or 'VALID')
        """
        with tf.variable_scope('conv'+str(counter)) as scope:
            w = self.create_weights([kernel_dim, kernel_dim,
                                     input_layer_depth, layer_depth])
            conv = tf.nn.conv2d(input_layer, w, strides=strides,
                                padding=padding)
            batched_conv = tf.layers.batch_normalization(conv, training=is_training)
            relu_conv = tf.nn.relu(batched_conv, name=scope.name)
            if self._monitoring >= 3:
                tf.summary.histogram("conv", conv)
                tf.summary.histogram("conv_activation", relu_conv)
            return relu_conv
    
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
        kernel_dim: integer
            Dimension `a` (in pixels) of the pooling kernel, defined as [1, a,
        a, 1] (only squared kernels are considered)
        stride: integer
            Dimension `a` (in pixels) of the pooling stride, defined as [1, a,
        a, 1], i.e. the shift between each pooling operation; we consider
        a regular shift (horizontal shift=vertical shift)
        padding: object
            String designing the padding mode ('SAME', or 'VALID')
        """
        with tf.variable_scope('pool' + str(counter)) as scope:
            return tf.nn.max_pool(input_layer,
                                  ksize=[1, kernel_dim, kernel_dim, 1],
                                  strides=[1,stride,stride,1], padding=padding)

    def fullyconnected_layer(self, counter, is_training, input_layer,
                             last_layer_dim, layer_depth, t_dropout=1.0):
        """Build a fully-connected layer as a tensor, into the convolutional
                       neural network

        Parameters
        ----------
        counter: integer
            fully-connected layer counter (for scope name unicity)
        is_training: tensor
            Boolean tensor that indicates the phase (training, or not); if training, batch
        normalization on batch statistics, otherwise batch normalization on population statistics
        (see `tf.layers.batch_normalization()` doc)
        input_layer: tensor
            Fully-connected layer input; output of the previous layer into the network
        last_layer_dim: integer
            previous layer depth, into the network
        layer_depth: integer
            full-connected layer depth
        t_dropout: tensor
            tensor corresponding to the neuron keeping probability during dropout operation
        """
        with tf.variable_scope('fc' + str(counter)) as scope:
            reshaped = tf.reshape(input_layer, [-1, last_layer_dim])
            w = self.create_weights([last_layer_dim, layer_depth])
            fc = tf.matmul(reshaped, w, name="raw_fc")
            batched_fc = tf.layers.batch_normalization(fc, training=is_training)
            relu_fc = tf.nn.relu(batched_fc, name="relu_fc")
            if self._monitoring >= 3:
                tf.summary.histogram("fc", fc)
                tf.summary.histogram("fc_activation", relu_fc)
            return tf.nn.dropout(relu_fc, t_dropout, name='relu_with_dropout')

    @abc.abstractmethod
    def compute_loss(self):
        """Define the loss tensor

        """
        return

    @abc.abstractmethod
    def optimize(self):
        """Define the optimizer; it may use a decaying learning rate following the equation

        """
        return

    def add_summary(self, metric, name):
        """ Add a TensorFlow scalar summary to the parameter metric, in order to monitor it in TensorBoard

        Parameter
        ---------
        metric: tensor
            metric to monitor by adding a tf.Summary
        name: object
            string designing the name of the metric (plotting purpose in TensorBoard)
        """
        summary = tf.summary.scalar(name, metric)
        metric_value, metric_update = tf.metrics.mean(metric, name='mean_' + name + '_op')
        mean_summary = tf.summary.scalar('mean_'+name, metric_value, collections=["update"])
        self._value_ops[name] = metric_value
        self._update_ops[name] = metric_update

    @abc.abstractmethod
    def define_batch(self, dataset, labels_of_interest, batch_size,
                     dataset_type="training"):
        """Insert images and labels in Tensorflow batches

        Parameters
        ----------
        dataset: Dataset
            Dataset that will feed the neural network; its `_image_size`
        attribute must correspond to those of this class
        labels_of_interest: list
            List of label indices on which a model will be trained
        batch_size: integer
            Number of images to set in each batch
        dataset_type: object
            string designing the considered dataset (`training`, `validation`
        or `testing`)
        """
        return

    @abc.abstractmethod
    def train(self, train_dataset, val_dataset, labels, keep_proba, nb_epochs,
              batch_size=20, validation_size=200, log_step=10, save_step=100,
              validation_step=200, nb_iter=None, backup_path=None,
              timing=False):
        """ Train the neural network on a specified dataset, during `nb_epochs`

        Parameters:
        -----------
        train_dataset: Dataset
            Dataset that will train the neural network; its `_image_size`
        attribute must correspond to those of this class
        validation_dataset: Dataset
            Validation dataset, to control the training process
        labels: list
            List of label indices on which a model will be trained
        keep_proba: float
            Probability of keeping a neuron during a training step (dropout)
        nb_epochs: integer
            Number of training epoch (one epoch=every image have been seen by
        the network); a larger value helps to reach higher
        accuracy, however the training time will be increased as well
        batch_size: integer
            Number of image per testing batchs
        log_step: integer
            Training process logging periodicity (quantity of iterations)
        save_step: integer
            Training process backing up periodicity (quantity of iterations)
        validation_step: integer
            Validation periodicity (quantity of iterations)
        nb_iter: integer
            Number of training iteration, overides nb_epochs if not None
        (mainly debogging purpose)
        backup_path: object
            String designing the place where must be saved the TensorFlow
        graph, summary and the model checkpoints
        timing: boolean
            If true, training phase execution time is measured and stored in a json file under
        <backup_path>/chronos/ repository

        """
        return

    @abc.abstractmethod
    def validate(self, batched_val_images, batched_val_labels, sess,
                 n_val_images, train_step, summary, writer):
        """ Validate the trained neural network on a validation dataset

        Parameters:
        -----------
        dataset: Dataset
            Dataset that will feed the neural network; its `_image_size`
        attribute must correspond to those of this class
        labels: list
        sess: tf.Session
        n_val_images: integer
        train_step: integer
        summary: tf.summary
        writer: tf.fileWriter

        """
        return

    @abc.abstractmethod
    def test(self, dataset, labels, batch_size=20, log_step=10, backup_path=None):
        """ Test the trained neural network on a testing dataset

        Parameters:
        -----------
        dataset: Dataset
            Dataset that will feed the neural network; its `_image_size`
        labels: list
            List of label indices on which a model will be trained
        batch_size: integer
            Number of image per testing batchs
        log_step: integer
            Training process logging periodicity (quantity of iterations)
        backup_path: object
            String designing the place where must be saved the TensorFlow
        graph, summary and the model checkpoints

        """
        return

    def summary(self):
        """ Print the network architecture on the command prompt
        """
        pass
