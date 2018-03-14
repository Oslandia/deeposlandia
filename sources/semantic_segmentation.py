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

import json
import math
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
import time

import dataset
from cnn_model import ConvolutionalNeuralNetwork
import utils

class SemanticSegmentationModel(ConvolutionalNeuralNetwork):

    def __init__(self, network_name="mapillary", image_size=512, nb_channels=3,
                 nb_labels=65, netsize="small", learning_rate=[1e-3],
                 monitoring_level=1):
        """
        """
        ConvolutionalNeuralNetwork.__init__(self, network_name, image_size, nb_channels, nb_labels, monitoring_level)
        self._Y = tf.placeholder(tf.int8, name='Y',
                                 shape=[None, self._image_size,
                                        self._image_size, self._nb_channels])
        self.add_layers()
        self.compute_loss()
        self.optimize(learning_rate)
        self._cm = self.compute_dashboard(self._Y, self._Y_pred)

    def add_layers(self):
        """
        """
        pass

    def compute_loss(self):
        """Define the loss tensor

        """
        pass

    def optimize(self):
        """Define the optimizer; it may use a decaying learning rate following the equation

        """
        pass

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
        pass

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
        pass

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
        pass

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
        pass
