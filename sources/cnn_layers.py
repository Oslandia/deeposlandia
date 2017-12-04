# Author: Raphael Delhome
# Organization: Oslandia
# Date: september 2017

import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops

import bpmll # Multilabel classification loss
import utils

def prepare_data(height, width, n_channels, batch_size,
                 labels_of_interest, datapath, dataset_type, scope_name):
    """Insert images and labels in Tensorflow batches

    Parameters
    ----------
    height: integer
        image height, in pixels
    width: integer
        image width, in pixels
    n_channels: integer
        Number of channels in the images (1 for grey-scaled images, 3 for RGB)
    batch_size: integer
        Size of the batchs, expressed as an image quantity
    labels_of_interest: list
        List of label indices on which a model will be trained
    datapath: object
        String designing the relative path to data
    dataset_type: object
        string designing the considered dataset (`training`, `validation` or `testing`)
    scope_name: object
        string designing the data preparation scope name
    
    """
    INPUT_PATH = os.path.join(datapath, dataset_type,
                              "input_" + str(width) + "_" + str(height))
    OUTPUT_PATH = os.path.join(datapath, dataset_type,
                              "output_" + str(width) + "_" + str(height))
    with tf.variable_scope(scope_name) as scope:
        # Reading image file paths
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
        image = tf.image.decode_jpeg(file_content, channels=n_channels)
        image.set_shape([height, width, n_channels])
        image = tf.div(image, 255) # Data normalization
        label = input_queue[1]
        # Collect batches of images before processing
        return tf.train.batch([image, label, input_queue[0]],
                              batch_size=batch_size,
                              num_threads=4)

def conv_layer(input_layer, input_layer_depth, kernel_dim, layer_depth,
               conv_strides, counter, network_name):
    """Build a convolutional layer as a Tensorflow object, for a convolutional
               neural network 

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
        Dimensions of the convolution stride operation, defined as [1, a, a, 1]
    where a is the shift (in pixels) between each convolution operation
    counter: integer
        Convolutional layer counter (for scope name unicity)
    network_name: object
        string designing the network name (for scope name unicity)
    
    """
    with tf.variable_scope(network_name + '_conv' + str(counter)) as scope:
        # Create kernel variable
        kernel = tf.get_variable('kernel',
                                 [kernel_dim, kernel_dim,
                                  input_layer_depth, layer_depth],
                                 initializer=tf.truncated_normal_initializer())
        # Create biases variable
        biases = tf.get_variable('biases',
                                 [layer_depth],
                                 initializer=tf.constant_initializer(0.0))
        # Apply the image convolution
        conv = tf.nn.conv2d(input_layer, kernel, strides=conv_strides,
                            padding='SAME')
        # Apply relu on the sum of convolution output and biases
        return tf.nn.relu(tf.add(conv, biases), name=scope.name)

def maxpool_layer(input_layer, pool_ksize, pool_strides,
                  counter, network_name):
    """Build a max pooling layer as a Tensorflow object, into the convolutional
                  neural network 

    Parameters
    ----------
    input_layer: tensor
        Pooling layer input; output of the previous layer into the network
    pool_ksize: list
        Dimension of the pooling kernel, defined as [1, a, a, 1] where a is the
    main kernel dimension (in pixels)=
    pool_strides: list
        Dimension of the pooling stride, defined as [1, a, a, 1] where a is the
    shift between each pooling operation
    counter: integer
        Max pooling layer counter (for scope name unicity)
    network_name: object
        string designing the network name (for scope name unicity)
    
    """
    with tf.variable_scope(network_name + '_pool' + str(counter)) as scope:
        return tf.nn.max_pool(input_layer, ksize=pool_ksize,
                               strides=pool_strides, padding='SAME')

def layer_dim(height, width, layer_coefs, last_layer_depth):
    """Consider the current layer depth as the function of previous layer
    hyperparameters, so as to reshape it as a single dimension layer

    Parameters
    ----------
    height: integer
        image height, in pixels
    width: integer
        image width, in pixels
    layer_coefs: list
        list of previous layer hyperparameters, that have an impact on the
    current layer depth
    last_layer_depth: integer
        depth of the last layer in the network
    
    """
    new_height = int(height / np.prod(np.array(layer_coefs)[:,2]))
    new_width = int(width / np.prod(np.array(layer_coefs)[:,1]))
    return new_height * new_width * last_layer_depth

def fullconn_layer(input_layer, height, width, last_layer_dim,
                   fc_layer_depth, t_dropout, counter, network_name):
    """Build a fully-connected layer as a tensor, into the convolutional
                   neural network

    Parameters
    ----------
    input_layer: tensor
        Fully-connected layer input; output of the previous layer into the network
    height: integer
        image height, in pixels
    width: integer
        image width, in pixels
    last_layer_dim: integer
        previous layer depth, into the network
    fc_layer_depth: integer
        full-connected layer depth
    t_dropout: tensor
        tensor corresponding to the neuron keeping probability during dropout operation
    counter: integer
        fully-connected layer counter (for scope name unicity)
    network_name: object
        string designing the network name (for scope name unicity)
    
    """
    with tf.variable_scope(network_name + '_fc' + str(counter)) as scope:
        reshaped = tf.reshape(input_layer, [-1, last_layer_dim])
        # Create weights and biases
        w = tf.get_variable('weights', [last_layer_dim, fc_layer_depth],
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [fc_layer_depth],
                            initializer=tf.constant_initializer(0.0))
        # Apply relu on matmul of reshaped and w + b
        fc = tf.nn.relu(tf.add(tf.matmul(reshaped, w), b), name='relu')
        # Apply dropout
        return tf.nn.dropout(fc, t_dropout, name='relu_with_dropout')

def output_layer(input_layer, input_layer_dim, n_output_classes, network_name):
    """Build an output layer to a neural network with a sigmoid final activation
    function (softmax if there is only one label to predict); return final
    network scores (logits) as well as predictions

    Parameters
    ----------
    input_layer: tensor
        Previous layer within the neural network (last hidden layer)
    input_layer_dim: integer
        Dimension of the previous neural network layer
    n_output_classes: integer
        Dimension of the output layer
    network_name: object
        String designing the network name (for scope name unicity)

    """
    # Output building
    with tf.variable_scope(network_name + '_output_layer') as scope:
        # Compute predicted outputs with softmax/sigmoid function
        if n_output_classes == 1:
            # Create weights and biases for the final fully-connected layer
            w = tf.get_variable('weights', [input_layer_dim, 2],
                                initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('biases', [2],
                                initializer=tf.random_normal_initializer())
            # Compute logits through a simple linear combination
            logits = tf.add(tf.matmul(input_layer, w), b)
            Y_raw_predict = tf.nn.softmax(logits)
            Y_predict = tf.to_int32(tf.round(tf.slice(Y_raw_predict, [0, 0], [-1, 1])))
            return logits, Y_raw_predict, Y_predict
        else:
            # Create weights and biases for the final fully-connected layer
            w = tf.get_variable('weights', [input_layer_dim, n_output_classes],
                                initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('biases', [n_output_classes],
                                initializer=tf.random_normal_initializer())
            # Compute logits through a simple linear combination
            logits = tf.add(tf.matmul(input_layer, w), b)
            Y_raw_predict = tf.nn.sigmoid(logits)
            Y_predict = tf.to_int32(tf.round(Y_raw_predict))
            return logits, Y_raw_predict, Y_predict

def define_loss(y_true, logits, y_raw_p, weights,
                start_lr, decay_steps, decay_rate, network_name):
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
    weights: tensor
        Values associated to each label for weighting loss contributions with
    respect to label popularity
    start_lr: integer
        Starting learning rate, used in the first iteration
    decay_steps: integer
        Number of steps over which the learning rate is computed
    decay_rate: float
        Decreasing rate used for learning rate computation
    network_name: object
        String designing the network name (for scope name unicity)
    """

    with tf.name_scope(network_name + '_loss'):
        if y_true.shape[1] == 1: # Mono-label: softmax
            # not_ytrue = tf.cast(tf.logical_not(tf.cast(y_true, tf.bool)), tf.float32)
            # Y_true_soft = tf.transpose(tf.reshape(tf.concat([y_true, not_ytrue], axis=0), [2, -1]))
            soft_logits = tf.slice(logits, [0, 0], [-1, 1])
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                              logits=soft_logits)
            bpmll_loss = tf.constant(0.0)
        else:
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                              logits=logits)
            bpmll_loss = bpmll.bp_mll_loss(y_true, y_raw_p)
        weighted_entropy = tf.multiply(weights, entropy)
        loss = tf.reduce_mean(weighted_entropy, name="loss")
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                                  name='global_step')
        lrate = tf.train.exponential_decay(start_lr, global_step,
                                           decay_steps=decay_steps,
                                           decay_rate=decay_rate,
                                           name='learning_rate')
        optimizer = tf.train.AdamOptimizer(lrate).minimize(loss, global_step)
        return {"loss": loss, "bpmll": bpmll_loss, "gs": global_step,
                "lrate": lrate, "optim": optimizer}

def convnet_building(X, param, img_width, img_height, nb_channels,
                     nb_labels, dropout,
                     network_name, nb_convpool, nb_fullconn):
    """Build the structure of a convolutional neural network from image data X
    to the last hidden layer, this layer being returned by this method  

    Parameters
    ----------
    X: tensorflow.placeholder
        Image data with a shape [batch_size, width, height, nb_channels]
    param: dict
        A dictionary of every network parameters (kernel sizes, strides, depths
    for each layer); the keys are the different layer, under the format
    <conv/pool/fullconn><rank>, e.g. conv1 for the first convolutional layer
    img_width: integer
        number of horizontal pixels within the image, i.e. first dimension
    img_height: integer
        number of vertical pixels within the image, i.e. second dimension
    nb_channels: integer
        number of channels within images, i.e. 1 if black and white, 3 if RGB
    images
    nb_labels: integer
        number of output classes (labels)
    dropout: tensor
        Represent the proportion of kept neurons within fully-connected network
    (to avoid over-fitting, some of them are deactivated at each iteration)
    network_name: object
        string designing the network name, for layer identification purpose
    nb_convpool: integer
        number of convolutional layer to add to the network (the number of
    pooling layer is the same, as one pooling layer is set after each
    convolutional layer)
    nb_fullconn: integer
        number of fully-connected layer to add to the network
    """
    layer_coefs = []

    i = 1
    while i <= nb_convpool:
        if i == 1:
            conv = conv_layer(X,
                              nb_channels,
                              param["conv"+str(i)]["kernel_size"],
                              param["conv"+str(i)]["depth"],
                              param["conv"+str(i)]["strides"],
                              i, network_name)
        else:
            conv = conv_layer(last_pool,
                              param["conv"+str(i-1)]["depth"],
                              param["conv"+str(i)]["kernel_size"],
                              param["conv"+str(i)]["depth"],
                              param["conv"+str(i)]["strides"],
                              i, network_name)
        layer_coefs.append(param["conv"+str(i)]["strides"])
        last_pool = maxpool_layer(conv,
                                  param["pool"+str(i)]["kernel_size"],
                                  param["pool"+str(i)]["strides"],
                                  i, network_name)
        last_layer_dim = param["conv"+str(i)]["depth"]
        layer_coefs.append(param["pool2"]["strides"])
        i = i + 1
                
    hidden_layer_dim = layer_dim(img_height, img_width,
                                 layer_coefs, last_layer_dim)
    
    i = 1
    while i <= nb_fullconn:
        if i == 1:
            last_fc = fullconn_layer(last_pool, img_height, img_width,
                                     hidden_layer_dim,
                                     param["fullconn1"]["depth"],
                                     dropout, i, network_name)
        else:
            last_fc = fullconn_layer(last_fc, img_height, img_width,
                                     param["fullconn"+str(i-1)]["depth"],
                                     param["fullconn"+str(i)]["depth"],
                                     dropout, i, network_name)
        last_fc_layer_dim = param["fullconn"+str(i)]["depth"]
        i = i + 1

    return output_layer(last_fc, last_fc_layer_dim, nb_labels, network_name)
