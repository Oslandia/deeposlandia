"""Unit test associated to the simple layer creation
"""

import keras as K

from deeposlandia.network import ConvolutionalNeuralNetwork

def test_convolution_shape():
    """Test the convolution operation through its output layer shape

    """
    image_size = 64
    depth = 8
    strides = 2
    cnn = ConvolutionalNeuralNetwork("test", image_size)
    y = cnn.convolution(cnn._X, nb_filters=depth,
                        kernel_size=3, strides=strides)
    assert(len(y.shape) == 4)
    assert(y.shape[1] == image_size // strides)
    assert(y.shape[2] == image_size // strides)
    assert(y.shape[3] == depth)

def test_transposed_convolution_shape():
    """Test the transposed convolution operation through its output layer shape

    """
    image_size = 64
    depth = 8
    kernel_size = 3
    strides = 2
    cnn = ConvolutionalNeuralNetwork("test", image_size)
    print(cnn._X.shape)
    y = cnn.transposed_convolution(cnn._X, nb_filters=depth,
                                   kernel_size=kernel_size, strides=strides)
    print(y.shape)
    assert(len(y.shape) == 4)
    assert(y.shape[1] == image_size // strides)
    assert(y.shape[2] == image_size // strides)
    assert(y.shape[3] == depth)

def test_maxpooling_shape():
    """Test the max pooling operation through its output layer shape

    """
    image_size = 64
    nb_channels = 3
    psize = 2
    strides = 2
    cnn = ConvolutionalNeuralNetwork("test", image_size, nb_channels)
    y = cnn.maxpool(cnn._X, pool_size=psize, strides=strides)
    assert(len(y.shape) == 4)
    assert(y.shape[1] == image_size // strides)
    assert(y.shape[2] == image_size // strides)
    assert(y.shape[3] == nb_channels)

def test_dense_shape():
    """Test the fully-connected layer through its output shape

    """
    image_size = 64
    depth = 8
    cnn = ConvolutionalNeuralNetwork("test", image_size)
    y = cnn.dense(cnn._X, depth=depth)
    assert(len(y.shape) == 2)
    assert(y.shape[1] == depth)
    
