# Raphael Delhome - september 2017

import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops

def prepare_data(height, width, n_channels, batch_size, dataset_type, scope_name):
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

# Convolutional layer
def conv_layer(input_layer, input_layer_depth, kernel_dim, layer_depth,
               conv_strides, counter):
    with tf.variable_scope('conv' + str(counter)) as scope:
        # Create kernel variable of dimension [K_C1, K_C1, NUM_CHANNELS, L_C1]
        kernel = tf.get_variable('kernel',
                                 [kernel_dim, kernel_dim,
                                  input_layer_depth, layer_depth],
                                 initializer=tf.truncated_normal_initializer())
        # Create biases variable of dimension [L_C1]
        biases = tf.get_variable('biases',
                                 [layer_depth],
                                 initializer=tf.constant_initializer(0.0))
        # Apply the image convolution
        conv = tf.nn.conv2d(input_layer, kernel, strides=conv_strides,
                            padding='SAME')
        # Apply relu on the sum of convolution output and biases
        # Output is of dimension BATCH_SIZE * IMAGE_HEIGHT * IMAGE_WIDTH * L_C1.
        return tf.nn.relu(tf.add(conv, biases), name=scope.name)

# Max-pooling layer
def maxpool_layer(input_layer, pool_ksize, pool_strides, counter):
    with tf.variable_scope('pool' + str(counter)) as scope:
        return tf.nn.max_pool(input_layer, ksize=pool_ksize,
                               strides=pool_strides, padding='SAME')
        # Output is of dimension BATCH_SIZE x 612 x 816 x L_C1

# Fully-connected layer
def layer_dim(height, width, layer_coefs, last_layer_depth):
    new_height = int(height / np.prod(np.array(layer_coefs)[:,2]))
    new_width = int(width / np.prod(np.array(layer_coefs)[:,1]))
    return new_height * new_width * last_layer_depth

def fullconn_layer(input_layer, height, width, last_layer_dim,
                   fc_layer_depth, t_dropout, counter):
    with tf.variable_scope('fc' + str(counter)) as scope:
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
