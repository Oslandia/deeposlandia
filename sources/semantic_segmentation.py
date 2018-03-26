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
        ConvolutionalNeuralNetwork.__init__(self, network_name, image_size, nb_channels,
                                            nb_labels, learning_rate, monitoring_level)
        self._Y = tf.placeholder(tf.int8, name='Y',
                                 shape=[None, self._image_size,
                                        self._image_size, self.get_nb_labels()])
        self.add_layers()
        self.compute_loss()
        self.optimize()
        self._cm = self.compute_dashboard(self._Y, self._Y_pred)

    def output_layer(self, input_layer):
        """Build an output layer to a neural network with a sigmoid final
        activation function; return final network scores (logits) as well
        as predictions

        Parameters
        ----------
        input_layer: tensor
            Previous layer within the neural network (last hidden layer)
        input_layer_dim: integer
            Dimension of the previous neural network layer
        """
        with tf.variable_scope('output_layer') as scope:
            self._logits = input_layer
            self._Y_raw_predict = tf.nn.softmax(self._logits, name="y_pred_raw")
            self._Y_pred = tf.round(self._Y_raw_predict, name="y_pred")

    def add_layers(self):
        """ Add layer to the neural network so as to answer the semantic segmentation problem

        """
        # Encoding
        layer = self.convolutional_layer(1, self._is_training, self._X, self._nb_channels, 3, 32)
        layer = self.maxpooling_layer(1, layer, 2, 2)
        layer = self.convolutional_layer(2, self._is_training, layer, 32, 3, 64)
        layer = self.maxpooling_layer(2, layer, 2, 2)
        layer = self.convolutional_layer(3, self._is_training, layer, 64, 3, 128)
        layer = self.maxpooling_layer(3, layer, 2, 2)
        layer = self.convolutional_layer(4, self._is_training, layer, 128, 3, 256)
        layer = self.maxpooling_layer(4, layer, 2, 2)
        # Decoding
        layer = self.convolution_transposal_layer(1, self._is_training, layer, 256,
                                                  3, 128, [1, 2, 2, 1])
        layer = self.convolution_transposal_layer(2, self._is_training, layer, 128,
                                                  3, 64, [1, 2, 2, 1])
        layer = self.convolution_transposal_layer(3, self._is_training, layer, 64,
                                                  3, 32, [1, 2, 2, 1])
        layer = self.convolution_transposal_layer(4, self._is_training, layer, 32, 3,
                                                  self.get_nb_labels(), [1, 2, 2, 1])
        return self.output_layer(layer)

    def compute_loss(self):
        """Define the loss tensor

        Warning: `softmax_cross_entropy_with_logits` is deprecated. Need for considering
        `softmax_cross_entropy_with_logits_v2` or an alternative way of doing in a future version

        """
        with tf.name_scope('loss'):
            self._entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._Y,
                                                                       logits=self._logits)
            if self._monitoring >= 3:
                tf.summary.histogram('xent', self._entropy)
            self._loss = tf.reduce_mean(self._entropy, name="mean_entropy")
            if self._monitoring >= 1:
                self.add_summary(self._loss, "loss")

    def optimize(self):
        """Define the optimizer; it may use a decaying learning rate following the equation

        """
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            opt = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        self._optimizer = opt.minimize(self._loss, self._global_step)

    def compute_dashboard(self, y_true, y_pred, label="wrapper"):
        """Compute the global confusion matrix (`y_true` and `y_pred`
        considered as one-dimension arrays) and the per-label confusion matrix;
        recursive algorithm that decomposes a 2D-array in multiple 1D-array;
        return a [1, -1]-shaped array

        Parameters:
        -----------
        y_true: tensor
            True values of y, 1D- or 2D-array (if 2D, trigger a recursive call)
        y_pred: tensor
            Predicted values of y, 1D- or 2D-array (if 2D, trigger a recursive
        call)
        label: object
            String designing the label for which confusion matrix is computed:
        "wrapper" for 2D-array calls (default value), either "global" or
        "labelX" for 1D-array calls
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
        with tf.variable_scope(dataset_type + "_data_pipe") as scope:
            image_filepaths = [dataset.image_info[i]["image_filename"]
                               for i in range(dataset.get_nb_images())]
            image_tensors = ops.convert_to_tensor(image_filepaths, dtype=tf.string,
                                                  name=dataset_type+"_images")
            label_filepaths = [dataset.image_info[i]["label_filename"]
                               for i in range(dataset.get_nb_images())]
            label_tensors = ops.convert_to_tensor(image_filepaths, dtype=tf.string,
                                                  name=dataset_type+"_labels")
            input_queue = tf.train.slice_input_producer([image_tensors,
                                                         label_tensors],
                                                        shuffle=True)
            image_content = tf.read_file(input_queue[0])
            images = tf.image.decode_image(image_content,
                                           channels=self._nb_channels)
            images.set_shape([self._image_size,
                             self._image_size,
                             self._nb_channels])
            label_content = tf.read_file(input_queue[1])
            labels = tf.image.decode_image(label_content, channels=self._nb_channels)
            labels.set_shape([self._image_size,
                              self._image_size,
                              self._nb_channels])
            #clean_labels = labels
            clean_labels = tf.one_hot(labels, len(labels_of_interest))
            #clean_labels = utils.to_categorical_tensor(labels, labels_of_interest)
            return tf.train.batch([images, clean_labels],
                                  batch_size=batch_size,
                                  num_threads=4)

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
        # If backup_path is undefined, set it with the dataset image path
        if backup_path == None:
            example_filename = train_dataset.image_info[0]['raw_filename']
            backup_path = "/".join(example_filename.split("/")[:2])
        # Define image batchs
        batched_images, batched_labels = self.define_batch(train_dataset, labels,
                                                           batch_size, "training")
        batched_val_images, batched_val_labels = self.define_batch(val_dataset, labels,
                                                                   validation_size, "validation")
        # Set up merged summary
        summary = tf.summary.merge_all()
        # Create tensorflow graph
        graph_path = os.path.join(backup_path, 'graph', self._network_name)
        train_writer = tf.summary.FileWriter(os.path.join(graph_path, "training"))
        val_writer = tf.summary.FileWriter(os.path.join(graph_path, "validation"))
        # Create folders to store checkpoints
        saver = tf.train.Saver(max_to_keep=1)
        ckpt_path = os.path.join(backup_path, 'checkpoints',
                                 self._network_name)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        os.makedirs(ckpt_path, exist_ok=True)
        ckpt = tf.train.get_checkpoint_state(ckpt_path)

        # Open a TensorFlow session to train the model with the batched dataset
        with tf.Session() as sess:
            train_writer.add_graph(sess.graph)
            val_writer.add_graph(sess.graph)
            # Initialize TensorFlow variables
            sess.run(tf.global_variables_initializer()) # training variable
            sess.run(tf.local_variables_initializer()) # validation variable (values, ops)
            # If checkpoint exists, restore model from it
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                utils.logger.info(("Recover model state from {}"
                                   "").format(ckpt.model_checkpoint_path))
            # Open a thread coordinator to use TensorFlow batching process
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            # Train the model
            start_time = time.time()
            if nb_iter is None:
                n_batches = int(len(train_dataset.image_info) / batch_size)
                nb_iter = n_batches * nb_epochs
            initial_step = self._global_step.eval(session=sess)
            for step in range(initial_step, nb_iter):
                X_batch, Y_batch = sess.run([batched_images, batched_labels])
                train_fd = {self._X: X_batch, self._Y: Y_batch,
                            self._dropout: keep_proba, self._is_training: True,
                            self._batch_size: batch_size}
                start_backp = time.time()
                sess.run(self._optimizer, feed_dict=train_fd)
                end_backp = time.time()
                if timing:
                    self._chrono["backprop"][step] = end_backp - start_backp
                if (step + 1) % log_step == 0 or step == initial_step:
                    start_monit = time.time()
                    s, loss, cm = sess.run([summary, self._loss, self._cm],
                                           feed_dict=train_fd)
                    train_writer.add_summary(s, step)
                    utils.logger.info(("step: {}, loss={:5.4f}, cm=[{:1.2f}, "
                                       "{:1.2f}, {:1.2f}, {:1.2f}]"
                                       "").format(step, loss, cm[0,0],
                                                  cm[0,1], cm[0,2], cm[0,3]))
                    end_monit = time.time()
                    if timing:
                        self._chrono["monitoring"][step] = end_monit - start_monit
                if (step + 1) % validation_step == 0:
                    start_valid = time.time()
                    self.validate(batched_val_images, batched_val_labels, sess,
                                  validation_size, step + 1, summary, val_writer)
                    end_valid = time.time()
                    if timing:
                        self._chrono["validation"][step] = end_valid - start_valid
                if (step + 1) % save_step == 0:
                    start_backup = time.time()
                    save_path = os.path.join(backup_path, 'checkpoints',
                                             self._network_name, 'step')
                    utils.logger.info(("Checkpoint {}-{} creation"
                                       "").format(save_path, step))
                    saver.save(sess, global_step=step, save_path=save_path)
                    end_backup = time.time()
                    if timing:
                        self._chrono["backup"][step] = end_backup - start_backup
            end_time = time.time()
            total_time = end_time - start_time
            utils.logger.info(("Optimization Finished! Total time: {:.2f} "
                               "seconds").format(total_time))
            if timing:
                self._chrono["total"] = total_time
            # Stop the thread coordinator
            coord.request_stop()
            coord.join(threads)
            if timing:
                chrono_dir = os.path.join(backup_path, "chronos")
                os.makedirs(chrono_dir, exist_ok=True)
                chrono_file = os.path.join(chrono_dir,
                                           self._network_name+"_chrono.json")
                with open(chrono_file, 'w') as written_file:
                    json.dump(self._chrono, fp=written_file, indent=4)
            # If checkpoint exists, restore model from it
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                utils.logger.info(("Recover model state from {}"
                                   "").format(ckpt.model_checkpoint_path))
            # Open a thread coordinator to use TensorFlow batching process
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            # Train the model
            start_time = time.time()
            if nb_iter is None:
                n_batches = int(len(train_dataset.image_info) / batch_size)
                nb_iter = n_batches * nb_epochs
            initial_step = self._global_step.eval(session=sess)
            for step in range(initial_step, nb_iter):
                X_batch, Y_batch = sess.run([batched_images, batched_labels])
                train_fd = {self._X: X_batch, self._Y: Y_batch,
                            self._dropout: keep_proba, self._is_training: True,
                            self._batch_size: batch_size}
                start_backp = time.time()
                sess.run(self._optimizer, feed_dict=train_fd)
                end_backp = time.time()
                if timing:
                    self._chrono["backprop"][step] = end_backp - start_backp
                if (step + 1) % log_step == 0 or step == initial_step:
                    start_monit = time.time()
                    s, loss, cm = sess.run([summary, self._loss, self._cm],
                                           feed_dict=train_fd)
                    train_writer.add_summary(s, step)
                    utils.logger.info(("step: {}, loss={:5.4f}, cm=[{:1.2f}, "
                                       "{:1.2f}, {:1.2f}, {:1.2f}]"
                                       "").format(step, loss, cm[0,0],
                                                  cm[0,1], cm[0,2], cm[0,3]))
                    end_monit = time.time()
                    if timing:
                        self._chrono["monitoring"][step] = end_monit - start_monit
                if (step + 1) % validation_step == 0:
                    start_valid = time.time()
                    self.validate(batched_val_images, batched_val_labels, sess,
                                  validation_size, step + 1, summary, val_writer)
                    end_valid = time.time()
                    if timing:
                        self._chrono["validation"][step] = end_valid - start_valid
                if (step + 1) % save_step == 0:
                    start_backup = time.time()
                    save_path = os.path.join(backup_path, 'checkpoints',
                                             self._network_name, 'step')
                    utils.logger.info(("Checkpoint {}-{} creation"
                                       "").format(save_path, step))
                    saver.save(sess, global_step=step, save_path=save_path)
                    end_backup = time.time()
                    if timing:
                        self._chrono["backup"][step] = end_backup - start_backup
            end_time = time.time()
            total_time = end_time - start_time
            utils.logger.info(("Optimization Finished! Total time: {:.2f} "
                               "seconds").format(total_time))
            if timing:
                self._chrono["total"] = total_time
            # Stop the thread coordinator
            coord.request_stop()
            coord.join(threads)
            if timing:
                chrono_dir = os.path.join(backup_path, "chronos")
                os.makedirs(chrono_dir, exist_ok=True)
                chrono_file = os.path.join(chrono_dir,
                                           self._network_name+"_chrono.json")
                with open(chrono_file, 'w') as written_file:
                    json.dump(self._chrono, fp=written_file, indent=4)


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
