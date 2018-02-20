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
                 batch_size=20, val_batch_size=100, nb_labels=65, learning_rate=1e-3):
        """ Class constructor
        """
        self._network_name = network_name
        self._image_size = image_size
        self._nb_channels = nb_channels
        self._nb_labels = nb_labels
        self._batch_size = batch_size
        self._val_batch_size = val_batch_size
        self._learning_rate = learning_rate
        self._value_ops = {}
        self._update_ops = {}

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

    def get_batch_size(self):
        """ `_batch_size` getter
        """
        return self._batch_size
    
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

    def convolutional_layer(self, counter, input_layer,
                            input_layer_depth, kernel_dim,
                            layer_depth, strides=[1, 1, 1, 1], padding='SAME'):
        """Build a convolutional layer as a Tensorflow object,
        for a convolutional neural network

        Parameters
        ----------
        counter: integer
            Convolutional layer counter (for scope name unicity)
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
            b = self.create_biases([layer_depth])
            conv = tf.nn.conv2d(input_layer, w, strides=strides,
                                padding=padding)
            tf.summary.histogram("conv", conv)
            relu_conv = tf.nn.relu(tf.add(conv, b), name=scope.name)
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

    def fullyconnected_layer(self, counter, input_layer,
                             last_layer_dim, layer_depth, t_dropout=1.0):
        """Build a fully-connected layer as a tensor, into the convolutional
                       neural network

        Parameters
        ----------
        counter: integer
            fully-connected layer counter (for scope name unicity)
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
            b = self.create_biases([layer_depth])
            fc = tf.add(tf.matmul(reshaped, w), b, name="raw_fc")
            tf.summary.histogram("fc", fc)
            relu_fc = tf.nn.relu(fc, name="relu_fc")
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
            logits = tf.add(tf.matmul(input_layer, w), b, name="logits")
            Y_raw_predict = tf.nn.sigmoid(logits, name="y_pred_raw")
            Y_pred = tf.round(Y_raw_predict, name="y_pred")
            tf.summary.histogram("logits", logits)
            tf.summary.histogram("y_raw_pred", Y_raw_predict)
            tf.summary.histogram("y_pred", Y_pred)
            return {"logits": logits, "y_raw_pred": Y_raw_predict, "y_pred": Y_pred}

    def add_layers(self, X, dropout):
        """Build the structure of a convolutional neural network from image data X
        to the last hidden layer, this layer being returned by this method

        Parameters
        ----------
        X: tensor
            Image data with a shape [batch_size, width, height, nb_channels]
        dropout: tensor
            Probability of keeping a neuron during a training step (dropout)
        """
        tf.summary.histogram("input", X)
        tf.summary.image("input", X)
        layer = self.convolutional_layer(1, X, self._nb_channels, 4, 16)
        layer = self.maxpooling_layer(1, layer, 2, 2)
        layer = self.convolutional_layer(3, layer, 16, 4, 32)
        layer = self.maxpooling_layer(2, layer, 2, 2)
        last_layer_dim = self.get_last_conv_layer_dim(4, 32) # pool mult, depth
        layer = self.fullyconnected_layer(1, layer, last_layer_dim, 128, dropout)
        return self.output_layer(layer, 128)

    def compute_loss(self, y_true, logits):
        """Define the loss tensor as well as the optimizer; it uses a decaying
        learning rate following the equation

        Parameters
        ----------
        y_true: tensor
            True labels (1 if the i-th label is true for j-th image, 0 otherwise)
        logits: tensor
            Logits computed by the model (scores associated to each labels for a
        given image)
        """
        with tf.name_scope(self._network_name + '_loss'):
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                              logits=logits)
            tf.summary.histogram('xent', entropy)
            mean_entropy = tf.reduce_mean(entropy, name="mean_entropy")
            self.add_summary(mean_entropy, "loss")
            return entropy, mean_entropy

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
        if len(self._learning_rate) == 1:
            lr = self._learning_rate
            opt = tf.train.AdamOptimizer(learning_rate=lr)
        else:
            lr = tf.train.exponential_decay(self._learning_rate[0],
                                            global_step,
                                            decay_steps=self._learning_rate[1],
                                            decay_rate=self._learning_rate[2],
                                            name='learning_rate')
            tf.summary.scalar("learning_rate", lr)
            opt = tf.train.AdamOptimizer(learning_rate=lr)
        optimizer = opt.minimize(loss, global_step)
        return {"gs": global_step, "lrate": lr, "optim": optimizer}

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
        tn = cmat[0, 0]
        self.add_summary(tn, "tn_" + label)
        fp = cmat[0, 1]
        self.add_summary(fp, "fp_" + label)
        fn = cmat[1, 0]
        self.add_summary(fn, "fn_" + label)
        tp = cmat[1, 1]
        self.add_summary(tp, "tp_" + label)
        metrics = self.compute_metrics(tn, fp, fn, tp, label)
        return tf.reshape(cmat, [1, -1], name="reshaped_cmat")

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

    def build(self, X, Y, dropout):
        """ Build the convolutional neural network structure from input
        placeholders to loss function optimization

        Parameters:
        -----------
        X: tensor
            Neural network input
        Y: tensor
            Neural network output
        dropout: tensor
            Probability of keeping a neuron during a training step (dropout)
        """
        output = self.add_layers(X, dropout)
        entropy, loss = self.compute_loss(Y, output["logits"])
        result = self.optimize(loss)
        cm = self.compute_dashboard(Y, output["y_pred"])
        result.update({"entropy": entropy,
                       "loss": loss,
                       "logits": output["logits"],
                       "y_raw_pred": output["y_raw_pred"],
                       "y_pred": output["y_pred"],
                       "conf_mat": cm})
        return result

    def define_batch(self, dataset, labels_of_interest, dataset_type="training"):
        """Insert images and labels in Tensorflow batches

        Parameters
        ----------
        dataset: Dataset
            Dataset that will feed the neural network; its `_image_size`
        attribute must correspond to those of this class
        labels_of_interest: list
            List of label indices on which a model will be trained
        dataset_type: object
            string designing the considered dataset (`training`, `validation`
        or `testing`)
        """
        scope_name = self._network_name + "_" + dataset_type + "_data_pipe"
        if dataset_type == "training":
            batch_size = self._batch_size
        else:
            batch_size = self._val_batch_size
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
            tf.summary.histogram("image", norm_images)
            return tf.train.batch([norm_images, input_queue[1]],
                                  batch_size=batch_size,
                                  num_threads=4)

    def train(self, train_dataset, val_dataset, labels, keep_proba, nb_epochs,
              log_step=10, save_step=100, nb_iter=None, backup_path=None,
              validation_step=200):
        """ Train the neural network on a specified dataset, during `nb_epochs`

        Parameters:
        -----------
        train_dataset: Dataset
            Dataset that will train the neural network; its `_image_size`
        attribute must correspond to those of this class
        validation_dataset: Dataset
            Validation dataset, to control the training process
        label: list
            List of label indices on which a model will be trained
        keep_proba: float
            Probability of keeping a neuron during a training step (dropout)
        nb_epochs: integer
            Number of training epoch (one epoch=every image have been seen by
        the network); a larger value helps to reach higher
        accuracy, however the training time will be increased as well
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
        batched_images, batched_labels = self.define_batch(train_dataset, labels, "training")
        batched_val_images, batched_val_labels = self.define_batch(val_dataset, labels, "validation")
        # Define model inputs and build the network
        X = tf.placeholder(tf.float32, name='X',
                           shape=[None, self._image_size,
                                  self._image_size, self._nb_channels])
        Y = tf.placeholder(tf.float32, name='Y', shape=[None, self._nb_labels])
        dropout = tf.placeholder(tf.float32, name="dropout")
        output = self.build(X, Y, dropout)
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
        # If that checkpoint exists, restore from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            utils.logger.info(("Recover model state from {}"
                               "").format(ckpt.model_checkpoint_path))

        # Open a TensorFlow session to train the model with the batched dataset
        with tf.Session() as sess:
            train_writer.add_graph(sess.graph)
            val_writer.add_graph(sess.graph)
            # Initialize TensorFlow variables
            sess.run(tf.global_variables_initializer()) # training variable
            sess.run(tf.local_variables_initializer()) # validation variable (values, ops)
            # Open a thread coordinator to use TensorFlow batching process
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            # Train the model
            start_time = time.time()
            if nb_iter is None:
                n_batches = int(len(train_dataset.image_info) / self._batch_size)
                nb_iter = n_batches * nb_epochs
            initial_step = output["gs"].eval(session=sess)
            for step in range(initial_step, nb_iter):
                X_batch, Y_batch = sess.run([batched_images, batched_labels])
                fd = {X: X_batch, Y: Y_batch, dropout: keep_proba}
                sess.run(output["optim"], feed_dict=fd)
                if (step + 1) % log_step == 0 or step == initial_step:
                    s, loss, cm = sess.run([summary, output["loss"], output["conf_mat"]],
                                           feed_dict=fd)
                    train_writer.add_summary(s, step)
                    utils.logger.info(("step: {}, loss={:5.4f}, cm={}"
                                       "").format(step, loss, cm[0,:4]))
                if (step + 1) % validation_step == 0:
                    self.validate(batched_val_images, batched_val_labels, sess,
                                  X, Y, dropout, len(val_dataset.image_info),
                                  step + 1, update_summary, val_writer, output)
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

    def validate(self, batched_val_images, batched_val_labels, sess, X, Y, dropout, n_val_images,
                 train_step, summary, writer, output):
        """ Validate the trained neural network on a validation dataset

        Parameters:
        -----------
        dataset: Dataset
            Dataset that will feed the neural network; its `_image_size`
        attribute must correspond to those of this class
        labels: list
        sess: tf.Session
        X : tf.placeholder
        Y: tf.placeholder
        dropout: tf.placeholder
        n_val_images: integer
        train_step: integer
        summary: tf.summary
        writer: tf.fileWriter
        output: dict

        """
        n_batches = int(n_val_images / self._val_batch_size)
        for step in range(n_batches):
            sess.run(tf.local_variables_initializer())
            X_val_batch, Y_val_batch = sess.run([batched_val_images, batched_val_labels])
            fd = {X: X_val_batch, Y: Y_val_batch, dropout: 1.0}
            sess.run([self._update_ops["loss"], self._update_ops["tn_global"],
                      self._update_ops["fp_global"], self._update_ops["fn_global"],
                      self._update_ops["tp_global"]], feed_dict=fd)
        vloss, vtn, vfp, vfn, vtp, vsum = sess.run([self._value_ops["loss"],
                                                    self._value_ops["tn_global"],
                                                    self._value_ops["fp_global"],
                                                    self._value_ops["fn_global"],
                                                    self._value_ops["tp_global"],
                                                    summary])
        writer.add_summary(vsum, train_step)
        utils.logger.info(("(validation) step: {}, loss={:5.4f}, cm=[{},{},{},{}]"
                           "").format(train_step, vloss, vtn, vfp, vfn, vtp))
        
    def test(self, dataset):
        """ Test the trained neural network on a testing dataset

        Parameters:
        -----------
        dataset: Dataset
            Dataset that will feed the neural network; its `_image_size`
        attribute must correspond to those of this class
        """
        pass

    def summary(self):
        """ Print the network architecture on the command prompt
        """
        pass
