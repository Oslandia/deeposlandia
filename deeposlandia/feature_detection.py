
import json
import math
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
import time

from deeposlandia import dataset, utils
from deeposlandia.cnn_model import ConvolutionalNeuralNetwork

class FeatureDetectionModel(ConvolutionalNeuralNetwork):

    def __init__(self, network_name="mapillary", image_size=512, nb_channels=3,
                 nb_labels=65, netsize="small", learning_rate=[1e-3],
                 monitoring_level=1):
        """ Class constructor
        """
        ConvolutionalNeuralNetwork.__init__(self, network_name, image_size, nb_channels,
                                            nb_labels, learning_rate, monitoring_level)
        self._Y = tf.placeholder(tf.float32, name='Y',
                                 shape=[None, self._nb_labels])
        if netsize == "small":
            self.add_layers_3_1()
        elif netsize == "medium":
            self.add_layers_6_2()
        elif netsize == "vgg":
            self.add_vgg_layers()
        elif netsize == "inception":
            self.add_inception_layers()
        else:
            utils.logger.error("Unsupported network.")
            sys.exit(1)
        self.compute_loss()
        self.optimize()
        self._cm = self.compute_dashboard(self._Y, self._Y_pred)

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
        with tf.variable_scope('output_layer') as scope:
            w = self.create_weights([input_layer_dim, self._nb_labels])
            b = self.create_biases([self._nb_labels])
            self._logits = tf.add(tf.matmul(input_layer, w), b, name="logits")
            self._Y_raw_predict = tf.nn.sigmoid(self._logits, name="y_pred_raw")
            self._Y_pred = tf.round(self._Y_raw_predict, name="y_pred")
            if self._monitoring >= 3:
                tf.summary.histogram("logits", self._logits)
                tf.summary.histogram("y_raw_pred", self._Y_raw_predict)

    def add_layers_3_1(self):
        """Build the structure of a convolutional neural network from image data `input_layer`
        to the last hidden layer, this layer being returned by this method; build a neural network
        with 3 convolutional+pooling layers and 1 fully-connected layer

        """
        if self._monitoring >= 3:
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
        if self._monitoring >= 3:
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

    def add_vgg_layers(self):
        """Build the structure of a convolutional neural network from image data `input_layer`
        to the last hidden layer on the model of a similar manner than VGG-net (see Simonyan &
        Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition, arXiv
        technical report, 2014) ; not necessarily the *same* structure, as the input shape is not
        necessarily identical

        Returns
        -------
        tensor
            Output layer of the neural network, *i.e.* a 1 X 1 X nb_class structure that contains
        model predictions
        """
        layer = self.convolutional_layer(1, self._is_training, self._X, self._nb_channels, 3, 64)
        layer = self.maxpooling_layer(1, layer, 2, 2)
        layer = self.convolutional_layer(2, self._is_training, layer, 64, 3, 128)
        layer = self.maxpooling_layer(2, layer, 2, 2)
        layer = self.convolutional_layer(3, self._is_training, layer, 128, 3, 256)
        layer = self.convolutional_layer(4, self._is_training, layer, 256, 3, 256)
        layer = self.maxpooling_layer(3, layer, 2, 2)
        layer = self.convolutional_layer(5, self._is_training, layer, 256, 3, 512)
        layer = self.convolutional_layer(6, self._is_training, layer, 512, 3, 512)
        layer = self.maxpooling_layer(4, layer, 2, 2)
        layer = self.convolutional_layer(7, self._is_training, layer, 512, 3, 512)
        layer = self.convolutional_layer(8, self._is_training, layer, 512, 3, 512)
        layer = self.maxpooling_layer(5, layer, 2, 2)
        last_layer_dim = self.get_last_conv_layer_dim(32, 512)
        layer = self.fullyconnected_layer(1, self._is_training, layer, last_layer_dim, 1024, self._dropout)
        return self.output_layer(layer, 1024)

    def inception_block(self, counter, input_layer, input_depth, depth_1,
                        depth_3_reduce, depth_3, depth_5_reduce, depth_5, depth_pool):
        """Apply an Inception block (concatenation of convoluted inputs, see Szegedy et al, 2014)

        Concatenation of several filtered outputs:
        - 1*1 convoluted image
        - 1*1 and 3*3 convoluted images
        - 1*1 and 5*5 convoluted images
        - 3*3 max-pooled and 1*1 convoluted images

        Parameters
        ----------
        counter : integer
            Inception block ID
        input_layer : tensor
            Input layer that has to be transformed in the Inception block
        input_depth : integer
            Input layer depth
        depth_1 : integer
            Depth of the 1*1 convoluted output
        depth_3_reduce : integer
            Hidden layer depth, between 1*1 and 3*3 convolution
        depth_3 : integer
            Depth of the 3*3 convoluted output
        depth_5_reduce : integer
            Hidden layer depth, between 1*1 and 5*5 convolution
        depth_5 : integer
            Depth of the 5*5 convoluted output
        depth_pool : integer
            Depth of the max-pooled output (after 1*1 convolution)

        Returns
        -------
        tensor
            Output layer, after Inception block treatments
        """
        filter_1_1 = self.convolutional_layer("i"+str(counter)+"1", self._is_training, input_layer,
                                              input_depth, 1, depth_1)
        filter_3_3 = self.convolutional_layer("i"+str(counter)+"3a", self._is_training, input_layer,
                                              input_depth, 1, depth_3_reduce)
        filter_3_3 = self.convolutional_layer("i"+str(counter)+"3b", self._is_training, filter_3_3,
                                              depth_3_reduce, 3, depth_3)
        filter_5_5 = self.convolutional_layer("i"+str(counter)+"5a", self._is_training, input_layer,
                                              input_depth, 1, depth_5_reduce)
        filter_5_5 = self.convolutional_layer("i"+str(counter)+"5b", self._is_training, filter_5_5,
                                              depth_5_reduce, 5, depth_5)
        filter_pool = self.maxpooling_layer("i"+str(counter), input_layer, 3, 1)
        filter_pool = self.convolutional_layer("i"+str(counter)+"p", self._is_training,
                                               filter_pool, input_depth, 1, depth_pool)
        return tf.concat([filter_1_1, filter_3_3, filter_5_5, filter_pool], axis=3)

    def add_inception_layers(self):
        """Build the structure of a convolutional neural network from image data `input_layer`
        to the last hidden layer on the model of a similar manner than Inception networks (see
        Szegedy et al, Going Deeper with Convolutions, arXiv technical report, 2014) ; not
        necessarily the *same* structure, as the input shape is not necessarily identical

        Returns
        -------
        tensor
            Output layer of the neural network, *i.e.* a 1 X 1 X nb_class structure that contains
        model predictions
        """
        layer = self.convolutional_layer(1, self._is_training, self._X, self._nb_channels, 7, 64,
        2)
        layer = self.maxpooling_layer(1, layer, 3, 2)
        layer = self.convolutional_layer(2, self._is_training, layer, 64, 3, 192)
        layer = self.maxpooling_layer(2, layer, 3, 2)
        layer = self.inception_block('3a', layer, 192, 64, 96, 128, 16, 32, 32)
        layer = self.inception_block('3b', layer, 256, 128, 128, 192, 32, 96, 64)
        layer = self.maxpooling_layer(3, layer, 3, 2)
        layer = self.inception_block('4a', layer, 480, 192, 96, 208, 16, 48, 64)
        layer = self.inception_block('4b', layer, 512, 160, 112, 224, 24, 64, 64)
        layer = self.inception_block('4c', layer, 512, 128, 128, 256, 24, 64, 64)
        layer = self.inception_block('4d', layer, 512, 112, 144, 288, 32, 64, 64)
        layer = self.inception_block('4e', layer, 528, 256, 160, 320, 32, 128, 128)
        layer = self.maxpooling_layer(4, layer, 3, 2)
        layer = self.inception_block('5a', layer, 832, 256, 160, 320, 32, 128, 128)
        layer = self.inception_block('5b', layer, 832, 384, 192, 384, 48, 128, 128)
        layer = tf.nn.avg_pool(layer, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1],
                               padding="VALID", name="avg_pool")
        layer = tf.reshape(layer, [-1, 1024])
        layer = tf.nn.dropout(layer, self._dropout, name="final_dropout")
        return self.output_layer(layer, 1024)

    def compute_loss(self):
        """Define the loss tensor as well as the optimizer; it uses a decaying
        learning rate following the equation

        """
        with tf.name_scope('loss'):
            self._entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self._Y,
                                                                    logits=self._logits)
            if self._monitoring >= 3:
                tf.summary.histogram('xent', self._entropy)
            self._loss = tf.reduce_mean(self._entropy, name="mean_entropy")
            if self._monitoring >= 1:
                self.add_summary(self._loss, "loss")

    def optimize(self):
        """Define the loss tensor as well as the optimizer; it uses a decaying
        learning rate following the equation

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
        with tf.name_scope("dashboard_" + label):
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
        norm_cmat = self.normalize_cm(cmat, label)
        tn = norm_cmat[0, 0]
        fp = norm_cmat[0, 1]
        fn = norm_cmat[1, 0]
        tp = norm_cmat[1, 1]
        normresh_cmat = tf.reshape(norm_cmat, [1, -1], name="reshaped_cmat")
        if self._monitoring >= 1:
            self.add_summary(tn, "tn_" + label)
            self.add_summary(fp, "fp_" + label)
            self.add_summary(fn, "fn_" + label)
            self.add_summary(tp, "tp_" + label)
            metrics = self.compute_metrics(tn, fp, fn, tp, label)
            metrics = tf.reshape(metrics, shape=[1, -1])
            normresh_cmat = tf.concat([normresh_cmat, metrics], 1)
        return normresh_cmat

    def normalize_cm(self, confusion_matrix, label):
        """Normalize the confusion matrix tensor so as to get items comprised between 0 and 1

        Parameters
        ----------
        confusion_matrix: tensor
            Confusion matrix of shape [2, 2]
        label: object
            Reference to the label of interest (either `global` or `labelX`, with X between 0 and
        `self._nb_labels`)
        Return
        ------
        tensor
            Normalized confusion matrix (shape [2, 2])
        """
        if label == "global":
            normalizer = tf.multiply(self._nb_labels, self._batch_size)
        else:
            normalizer = self._batch_size
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
        acc = tf.divide(tf.add(tn, tp), tn + fp + fn + tp)
        self.add_summary(acc, "acc_"+label)
        if self._monitoring < 2:
            return [acc]
        else:
            pos_true = tf.add(tp, fn)
            neg_true = tf.add(fp, tn)
            pos_pred = tf.add(tp, fp)
            neg_pred = tf.add(tn, fn)
            tpr = tf.divide(tp, pos_true)
            fpr = tf.divide(fp, neg_true)
            tnr = tf.divide(tn, neg_true)
            fnr = tf.divide(fn, pos_true)
            ppv = tf.divide(tp, pos_pred)
            npv = tf.divide(tn, neg_pred)
            fm = 2.0 * tf.divide(tf.multiply(ppv, tpr), tf.add(ppv, tpr))
            self.add_summary(pos_true, "pos_true_"+label)
            self.add_summary(neg_true, "neg_true_"+label)
            self.add_summary(pos_pred, "pos_pred_"+label)
            self.add_summary(neg_pred, "neg_pred_"+label)
            self.add_summary(tpr, "tpr_"+label)
            self.add_summary(fpr, "fpr_"+label)
            self.add_summary(tnr, "tnr_"+label)
            self.add_summary(fnr, "fnr_"+label)
            self.add_summary(ppv, "ppv_"+label)
            self.add_summary(npv, "npv_"+label)
            self.add_summary(fm, "fm_"+label)
            return [acc, pos_pred, neg_pred, tpr, ppv, fm]

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
        with tf.variable_scope(dataset_type + "_data_pipe") as scope:
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
        graph_path = os.path.join(backup_path, 'logs', self._network_name)
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
        os.makedirs(os.path.dirname(result_dir), exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        y_raw_pred = np.zeros([dataset.get_nb_images(), self._nb_labels])
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
                y_raw_pred[step*batch_size:(step+1)*batch_size,:] = sess.run(self._Y_raw_predict, feed_dict=test_fd)
                y_pred[step*batch_size:(step+1)*batch_size,:] = sess.run(self._Y_pred, feed_dict=test_fd)
            utils.logger.info(("Inference finished! Total time: {:.2f} "
                               "seconds").format(time.time() - start_time))

            # Stop the thread coordinator
            coord.request_stop()
            coord.join(threads)

        label_name = [dataset.class_info[k]['name'] for k in range(dataset.get_nb_class())]
        label_occurrence = sum(y_pred)
        label_popularity = pd.DataFrame({"name": label_name, "occurrence": label_occurrence})
        utils.logger.info(("In the {} images of the testing set, the label occurrences "
                           "are as follows:").format(len(dataset.image_info)))
        utils.logger.info(label_popularity.sort_values("occurrence", ascending=False))
        # Build output structures
        prediction_keys = [dataset.image_info[k]['image_filename'] for k in
                           range(len(dataset.image_info))]
        predictions = dict(zip(prediction_keys, y_raw_pred.tolist()))
        result_dict = {"train_step": int(train_step),
                       "predictions": predictions}
        with open(os.path.join(result_dir, "inference_"+str(train_step)+".json"), "w") as f:
            json.dump(result_dict, f, allow_nan=True)
