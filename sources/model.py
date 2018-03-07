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
import utils

class ConvolutionalNeuralNetwork(object):

    def __init__(self, network_name="mapillary", image_size=512, nb_channels=3,
                 nb_labels=65, netsize="small", learning_rate=[1e-3]):
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
        self._Y = tf.placeholder(tf.float32, name='Y',
                                 shape=[None, self._nb_labels])
        self._dropout = tf.placeholder(tf.float32, name="dropout")
        self._is_training = tf.placeholder(tf.bool, name="is_training")
        self._batch_size = tf.placeholder(tf.int32, name="batch_size")
        if netsize == "small":
            self.add_layers_3_1()
        else:
            self.add_layers_6_2()
        self.compute_loss()
        self.optimize(learning_rate)
        self._cm = self.compute_dashboard(self._Y, self._Y_pred)

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
        with tf.variable_scope(self._network_name+'_conv'+str(counter)) as scope:
            w = self.create_weights([kernel_dim, kernel_dim,
                                     input_layer_depth, layer_depth])
            conv = tf.nn.conv2d(input_layer, w, strides=strides,
                                padding=padding)
            tf.summary.histogram("conv", conv)
            batched_conv = tf.layers.batch_normalization(conv, training=is_training)
            relu_conv = tf.nn.relu(batched_conv, name=scope.name)
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
        with tf.variable_scope(self._network_name + '_fc' + str(counter)) as scope:
            reshaped = tf.reshape(input_layer, [-1, last_layer_dim])
            w = self.create_weights([last_layer_dim, layer_depth])
            fc = tf.matmul(reshaped, w, name="raw_fc")
            tf.summary.histogram("fc", fc)
            batched_fc = tf.layers.batch_normalization(fc, training=is_training)
            relu_fc = tf.nn.relu(batched_fc, name="relu_fc")
            tf.summary.histogram("fc_activation", relu_fc)
            return tf.nn.dropout(relu_fc, t_dropout, name='relu_with_dropout')

    def output_layer(self, input_layer, input_layer_dim):
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
        with tf.variable_scope(self._network_name + '_output_layer') as scope:
            w = self.create_weights([input_layer_dim, self._nb_labels])
            b = self.create_biases([self._nb_labels])
            self._logits = tf.add(tf.matmul(input_layer, w), b, name="logits")
            self._Y_raw_predict = tf.nn.sigmoid(self._logits, name="y_pred_raw")
            self._Y_pred = tf.round(self._Y_raw_predict, name="y_pred")
            tf.summary.histogram("logits", self._logits)
            tf.summary.histogram("y_raw_pred", self._Y_raw_predict)

    def add_layers_3_1(self):
        """Build the structure of a convolutional neural network from image data `input_layer`
        to the last hidden layer, this layer being returned by this method; build a neural network
        with 3 convolutional+pooling layers and 1 fully-connected layer

        """
        tf.summary.histogram("input", self._X)
        tf.summary.image("input", self._X)
        layer = self.convolutional_layer(1, self._is_training, self._X, self._nb_channels, 7, 16)
        layer = self.maxpooling_layer(1, layer, 2, 2)
        layer = self.convolutional_layer(2, self._is_training, layer, 16, 5, 32)
        layer = self.maxpooling_layer(2, layer, 2, 2)
        layer = self.convolutional_layer(3, self._is_training, layer, 32, 3, 64)
        layer = self.maxpooling_layer(3, layer, 2, 2)
        last_layer_dim = self.get_last_conv_layer_dim(8, 64) # pool mult, depth
        layer = self.fullyconnected_layer(1, self._is_training, layer, last_layer_dim, 512, self._dropout)
        return self.output_layer(layer, 512)

    def add_layers_6_2(self):
        """Build the structure of a convolutional neural network from image data `input_layer`
        to the last hidden layer, this layer being returned by this method; build a neural network
        with 6 convolutional+pooling layers and 2 fully-connected layers

        """
        tf.summary.histogram("input", self._X)
        tf.summary.image("input", self._X)
        layer = self.convolutional_layer(1, self._is_training, self._X, self._nb_channels, 7, 16)
        layer = self.maxpooling_layer(1, layer, 2, 2)
        layer = self.convolutional_layer(2, self._is_training, layer, 16, 7, 32)
        layer = self.maxpooling_layer(2, layer, 2, 2)
        layer = self.convolutional_layer(3, self._is_training, layer, 32, 5, 64)
        layer = self.maxpooling_layer(3, layer, 2, 2)
        layer = self.convolutional_layer(4, self._is_training, layer, 64, 5, 128)
        layer = self.maxpooling_layer(4, layer, 2, 2)
        layer = self.convolutional_layer(5, self._is_training, layer, 128, 3, 256)
        layer = self.maxpooling_layer(5, layer, 2, 2)
        layer = self.convolutional_layer(6, self._is_training, layer, 256, 3, 256)
        layer = self.maxpooling_layer(6, layer, 2, 2)
        last_layer_dim = self.get_last_conv_layer_dim(64, 256) # pool mult, depth
        layer = self.fullyconnected_layer(1, self._is_training, layer, last_layer_dim, 1024, self._dropout)
        layer = self.fullyconnected_layer(2, self._is_training, layer, 1024, 512, self._dropout)
        return self.output_layer(layer, 512)

    def compute_loss(self):
        """Define the loss tensor as well as the optimizer; it uses a decaying
        learning rate following the equation

        """
        with tf.name_scope(self._network_name + '_loss'):
            self._entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self._Y,
                                                                    logits=self._logits)
            tf.summary.histogram('xent', self._entropy)
            self._loss = tf.reduce_mean(self._entropy, name="mean_entropy")
            self.add_summary(self._loss, "loss")

    def optimize(self, learning_rate):
        """Define the loss tensor as well as the optimizer; it uses a decaying
        learning rate following the equation

        Parameter
        ---------
        learning_rate: float or list
            Either a constant learning rate or a list of learning rate floating components
        (starting learning rate, decay step and decay rate)
        """
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
        with tf.name_scope(self._network_name + "_dashboard_" + label):
            if len(y_true.shape) > 1:
                yt_resh = tf.reshape(y_true, [-1], name="1D-y-true")
                yp_resh = tf.reshape(y_pred, [-1], name="1D-y-pred")
                cmat = self.compute_dashboard(yt_resh, yp_resh, "global")
                for i in range(y_true.shape[1]):
                    yti = yt_resh[i:yt_resh.shape[0]:y_true.shape[1]]
                    ypi = yp_resh[i:yp_resh.shape[0]:y_pred.shape[1]]
                    cmi = self.compute_dashboard(yti, ypi, "label"+str(i))
                    cmat = tf.concat([cmat, cmi], axis=1)
                return cmat
            else:
                return self.confusion_matrix(y_true, y_pred, label)

    def confusion_matrix(self, y_true, y_pred, label):
        """ Personnalized confusion matrix computing, that returns a [1,
        -1]-shaped tensor, to allow further concatenating operations

        Parameters:
        y_true: tensor
            True values of y, 1D-array
        y_pred: tensor
            Predicted values of y, 1D-array
        label: object
            String designing the label for which confusion matrix is computed:
        "wrapper" for 2D-array calls (default value), either "global" or
        "labelX" for 1D-array calls
        """
        cmat = tf.confusion_matrix(y_true, y_pred, num_classes=2, name="cmat")
        norm_cmat = self.normalize_cm(cmat)
        tn = norm_cmat[0, 0]
        self.add_summary(tn, "tn_" + label)
        fp = norm_cmat[0, 1]
        self.add_summary(fp, "fp_" + label)
        fn = norm_cmat[1, 0]
        self.add_summary(fn, "fn_" + label)
        tp = norm_cmat[1, 1]
        self.add_summary(tp, "tp_" + label)
        metrics = self.compute_metrics(tn, fp, fn, tp, label)
        return tf.reshape(norm_cmat, [1, -1], name="reshaped_cmat")

    def normalize_cm(self, confusion_matrix):
        """Normalize the confusion matrix tensor so as to get items comprised between 0 and 1

        :param confusion_matrix: tensor - confusion matrix of shape [2, 2]
        :return: tensor - normalized confusion matrix (shape [2, 2])
        """
        normalizer = tf.multiply(self._nb_labels, self._batch_size)
        return tf.divide(confusion_matrix, normalizer, "norm_cmat")

    def compute_metrics(self, tn, fp, fn, tp, label):
        """Compute a wide range of confusion-matrix-related metrics, such as
        accuracy, precision, recall and so on; create associated summaries for
        tensorboard monitoring

        Parameters:
        -----------
        tn: tensor
            Number of true negative predictions for current batch and label
        fp: tensor
            Number of false positive predictions for current batch and label
        fn: tensor
            Number of false negative predictions for current batch and label
        tp: tensor
            Number of true positive predictions for current batch and label
        label: object
            String designing the label for which confusion matrix is computed:
        "wrapper" for 2D-array calls (default value), either "global" or
        "labelX" for 1D-array calls
        """
        pos_true = tf.add(tp, fn)
        self.add_summary(pos_true, "pos_true_"+label)
        neg_true = tf.add(fp, tn)
        self.add_summary(neg_true, "neg_true_"+label)
        pos_pred = tf.add(tp, fp)
        self.add_summary(pos_pred, "pos_pred_"+label)
        neg_pred = tf.add(tn, fn)
        self.add_summary(neg_pred, "neg_pred_"+label)
        acc = tf.divide(tf.add(tn, tp), tn + fp + fn + tp)
        self.add_summary(acc, "acc_"+label)
        tpr = tf.divide(tp, tf.add(tp, fn))
        self.add_summary(tpr, "tpr_"+label)
        fpr = tf.divide(fp, tf.add(tn, fp))
        self.add_summary(fpr, "fpr_"+label)
        tnr = tf.divide(tn, tf.add(tn, fp))
        self.add_summary(tnr, "tnr_"+label)
        fnr = tf.divide(fn, tf.add(tp, fn))
        self.add_summary(fnr, "fnr_"+label)
        ppv = tf.divide(tp, tf.add(tp, fp))
        self.add_summary(ppv, "ppv_"+label)
        npv = tf.divide(tn, tf.add(tn, fn))
        self.add_summary(npv, "npv_"+label)
        fm = 2.0 * tf.divide(tf.multiply(ppv, tpr), tf.add(ppv, tpr))
        self.add_summary(fm, "fm_"+label)
        return [pos_pred, neg_pred, acc, tpr, ppv, fm]

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
        scope_name = self._network_name + "_" + dataset_type + "_data_pipe"
        with tf.variable_scope(scope_name) as scope:
            filepaths = [dataset.image_info[i]["image_filename"]
                         for i in range(dataset.get_nb_images())]
            filepath_tensors = ops.convert_to_tensor(filepaths, dtype=tf.string,
                                                  name=dataset_type+"_images")
            labels = [[dataset.image_info[i]["labels"][l]
                       for l in labels_of_interest]
                      for i in range(dataset.get_nb_images())]
            label_tensors = ops.convert_to_tensor(labels, dtype=tf.int16,
                                           name=dataset_type+"_labels")
            input_queue = tf.train.slice_input_producer([filepath_tensors,
                                                         label_tensors],
                                                        shuffle=True)
            file_content = tf.read_file(input_queue[0])
            images = tf.image.decode_image(file_content,
                                           channels=self._nb_channels)
            images.set_shape([self._image_size,
                             self._image_size,
                             self._nb_channels])
            norm_images = tf.image.per_image_standardization(images)
            return tf.train.batch([norm_images, input_queue[1]],
                                  batch_size=batch_size,
                                  num_threads=4)

    def train(self, train_dataset, val_dataset, labels, keep_proba, nb_epochs,
              batch_size=20, validation_size=200, log_step=10, save_step=100,
              validation_step=200, nb_iter=None, backup_path=None):
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
        nb_iter: integer
            Number of training iteration, overides nb_epochs if not None
        (mainly debogging purpose)
        backup_path: object
            String designing the place where must be saved the TensorFlow
        graph, summary and the model checkpoints
        validation_step: integer
            Validation periodicity (quantity of iterations)

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
        # Set up train and validation summaries
        summary = tf.summary.merge_all()
        update_summary = tf.summary.merge_all("update")
        # Create tensorflow graph
        graph_path = os.path.join(backup_path, 'graph', self._network_name)
        train_writer = tf.summary.FileWriter(os.path.join(graph_path, "training"))
        val_writer = tf.summary.FileWriter(os.path.join(graph_path, "validation"))
        # Create folders to store checkpoints
        saver = tf.train.Saver(max_to_keep=1)
        ckpt_path = os.path.join(backup_path, 'checkpoints',
                                 self._network_name)
        utils.make_dir(os.path.dirname(ckpt_path))
        utils.make_dir(ckpt_path)
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
                sess.run(self._optimizer, feed_dict=train_fd)
                if (step + 1) % log_step == 0 or step == initial_step:
                    s, loss, cm = sess.run([summary, self._loss, self._cm],
                                           feed_dict=train_fd)
                    train_writer.add_summary(s, step)
                    utils.logger.info(("step: {}, loss={:5.4f}, cm=[{:1.2f}, "
                                       "{:1.2f}, {:1.2f}, {:1.2f}]"
                                       "").format(step, loss, cm[0,0],
                                                  cm[0,1], cm[0,2], cm[0,3]))
                if (step + 1) % validation_step == 0:
                    self.validate(batched_val_images, batched_val_labels, sess,
                                  validation_size, step + 1, summary, val_writer)
                    # of update_summary
                if (step + 1) % save_step == 0:
                    save_path = os.path.join(backup_path, 'checkpoints',
                                             self._network_name, 'step')
                    utils.logger.info(("Checkpoint {}-{} creation"
                                       "").format(save_path, step))
                    saver.save(sess, global_step=step, save_path=save_path)
            utils.logger.info(("Optimization Finished! Total time: {:.2f} "
                               "seconds").format(time.time() - start_time))
            # Stop the thread coordinator
            coord.request_stop()
            coord.join(threads)

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
        X_val_batch, Y_val_batch = sess.run([batched_val_images, batched_val_labels])
        val_fd = {self._X: X_val_batch, self._Y: Y_val_batch,
                  self._dropout: 1.0, self._is_training: False,
                  self._batch_size: n_val_images}
        vloss, vcm, vsum = sess.run([self._loss, self._cm, summary], feed_dict=val_fd)
        writer.add_summary(vsum, train_step)
        utils.logger.info(("(validation) step: {}, loss={:5.4f}, cm=[{:1.2f}, "
                           "{:1.2f}, {:1.2f}, {:1.2f}]"
                           "").format(train_step, vloss, vcm[0,0],
                                      vcm[0,1], vcm[0,2], vcm[0,3]))

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
        # If backup_path is undefined, set it with the dataset image path
        if backup_path == None:
            example_filename = train_dataset.image_info[0]['raw_filename']
            backup_path = "/".join(example_filename.split("/")[:2])
        # Define image batchs
        batched_images, batched_labels = self.define_batch(dataset, labels, batch_size, "testing")
        # Create folders to store checkpoints and save inference results
        saver = tf.train.Saver(max_to_keep=1)
        ckpt_path = os.path.join(backup_path, 'checkpoints', self._network_name)
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        result_dir = os.path.join(backup_path, "results", self._network_name)
        utils.make_dir(os.path.dirname(result_dir))
        utils.make_dir(result_dir)

        y_pred = np.zeros([dataset.get_nb_images(), self._nb_labels])
        # Open a TensorFlow session to train the model with the batched dataset
        with tf.Session() as sess:
            # Initialize TensorFlow variables
            sess.run(tf.global_variables_initializer()) # training variables
            sess.run(tf.local_variables_initializer()) # testing variables (value, update ops)
            # If checkpoint exists, restore model from it
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                utils.logger.info(("Recover model state from {}"
                                   "").format(ckpt.model_checkpoint_path))
            else:
                utils.logger.warning("No trained model.")
            train_step = self._global_step.eval(session=sess)
            # Open a thread coordinator to use TensorFlow batching process
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            # Test the trained model
            start_time = time.time()
            n_batches = int(len(dataset.image_info) / batch_size)
            for step in range(n_batches):
                X_test_batch, Y_test_batch = sess.run([batched_images, batched_labels])
                test_fd = {self._X: X_test_batch, self._Y: Y_test_batch,
                            self._dropout: 1.0, self._is_training: False,
                            self._batch_size: batch_size}
                y_pred[step:step+batch_size,:] = sess.run(self._Y_raw_predict, feed_dict=test_fd)
                print(y_pred[step:step+batch_size,:])
                print(Y_test_batch)
                sess.run(list(self._update_ops.values()), feed_dict=test_fd)
                loss, cm = sess.run([self._loss, self._cm],
                                    feed_dict=test_fd)
                utils.logger.info(("step: {}, loss={:5.4f}, cm=[{:1.2f}, "
                                   "{:1.2f}, {:1.2f}, {:1.2f}]"
                                   "").format(step, loss, cm[0,0],
                                              cm[0,1], cm[0,2], cm[0,3]))
            result_dict = {"train_step": int(train_step),
                           "y_pred": y_pred.tolist()}
            test_dashboard = sess.run(list(self._value_ops.values()))
            result_dict.update(dict(zip(list(self._value_ops.keys()),
                                        [float(a) for a in test_dashboard])))
            utils.logger.info(("average cm=[{:1.2f}, {:1.2f}, {:1.2f}, {:1.2f}]"
                               "").format(result_dict["tn_global"], result_dict["fp_global"],
                                          result_dict["fn_global"], result_dict["tp_global"]))
            utils.logger.info(("Inference finished! Total time: {:.2f} "
                               "seconds").format(time.time() - start_time))
            # Stop the thread coordinator
            coord.request_stop()
            coord.join(threads)
            with open(os.path.join(result_dir, "inference_"+str(train_step)+".json"), "w") as f:
                json.dump(result_dict, f, allow_nan=True)

    def summary(self):
        """ Print the network architecture on the command prompt
        """
        pass
