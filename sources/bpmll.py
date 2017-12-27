# Copyright (c) 2017 Lukas Huwald

# You are free to use or modify this implementation for research and non-commercial purposes, as long as you include this license file.

# There is no warranty of any kind. The author is in no way responsible for any
# consequences resulting of the use of this implementation.

import numpy as np
import tensorflow as tf

def multilabel_loss(y_true, y_pred):
    """ Multi-label loss computing according to Vallet&Sakamoto (2015)

    Parameters
    ----------
    y_true: tensor
        2D tensor that represents the true labels
    y_pred: tensor
        2D tensor that represents the model return
    """
    # ml_num_term = [i for i,j in zip(y_pred, y_true) if j==1]
    ml_num_term = y_pred
    ml_loss_basis = tf.divide(tf.reduce_sum(tf.exp(ml_num_term)),
                              tf.reduce_sum(tf.exp(y_pred)))
    ml_loss = tf.reduce_mean(tf.negative(tf.log(ml_loss_basis)))
    return ml_loss

def bp_mll_loss(y_true, y_pred):
    """BP-MLL loss function; `y_true` and `y_pred` must be 2D tensors of shape
    (batch dimension, number of labels); `y_true` must satisfy `y_true[i][j] ==
    1` iff sample `i` has label `j`

    *cf* "Zhang, Min-Ling, and Zhi-Hua Zhou. "Multilabel neural networks with
     applications to functional genomics and text categorization." IEEE
     transactions on Knowledge and Data Engineering 18.10 (2006): 1338-1351."

    Parameters
    ----------
    y_true: tensor
        2D tensor that represents the true labels
    y_pred: tensor
        2D tensor that represents the predicted labels

    """
    # get true and false labels
    shape = tf.shape(y_true)
    y_i = tf.equal(y_true, tf.ones(shape))
    y_i_bar = tf.not_equal(y_true, tf.ones(shape))

    # get indices to check
    truth_matrix = tf.to_float(pairwise_and(y_i, y_i_bar))

    # calculate all exp'd differences
    sub_matrix = pairwise_sub(y_pred, y_pred)
    exp_matrix = tf.exp(tf.negative(sub_matrix))

    # check which differences to consider and sum them
    sparse_matrix = tf.multiply(exp_matrix, truth_matrix)
    sums = tf.reduce_sum(sparse_matrix, axis=[1,2])

    # get normalizing terms and apply them
    y_i_sizes = tf.reduce_sum(tf.to_float(y_i), axis=1)
    y_i_bar_sizes = tf.reduce_sum(tf.to_float(y_i_bar), axis=1)
    normalizers = tf.multiply(y_i_sizes, y_i_bar_sizes)
    results = tf.divide(sums, normalizers)

    # sum over samples
    return tf.reduce_sum(results)

def pairwise_sub(a, b):
    """Compute pairwise differences between elements of the tensors a and b

    Parameters
    ----------
    a: tensor
        first tensor to compare
    b: tensor
        second tensor to compare
    
    """
    column = tf.expand_dims(a, 2)
    row = tf.expand_dims(b, 1)
    return tf.subtract(column, row)

def pairwise_and(a, b):
    """Compute pairwise logical `and` between elements of the tensors a and b

    Parameters
    ----------
    a: tensor
        first tensor to compare
    b: tensor
        second tensor to compare
    
    """
    column = tf.expand_dims(a, 2)
    row = tf.expand_dims(b, 1)
    return tf.logical_and(column, row)
