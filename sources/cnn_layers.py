# Author: Raphael Delhome
# Organization: Oslandia
# Date: september 2017

import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops

def prepare_data(height, width, n_channels, batch_size,
                 dataset_type, scope_name):
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
    dataset_type: object
        string designing the considered dataset (`training`, `validation` or `testing`)
    scope_name: object
        string designing the data preparation scope name
    
    """
    INPUT_PATH = os.path.join("..", "data", dataset_type, "input")
    OUTPUT_PATH = os.path.join("..", "data", dataset_type, "output")
    with tf.variable_scope(scope_name) as scope:
        # Reading image file paths
        filepaths = os.listdir(INPUT_PATH)
        filepaths.sort()
        filepaths = [os.path.join(INPUT_PATH, fp) for fp in filepaths]
        images = ops.convert_to_tensor(filepaths, dtype=tf.string,
                                       name=dataset_type+"_images")
        # Reading labels
        labels = (pd.read_csv(os.path.join(OUTPUT_PATH, "labels.csv"))
                  .iloc[:,6:].values)
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
