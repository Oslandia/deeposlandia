"""Unit test related to the simple layer creation
"""

from keras.models import Model

from deeposlandia.network import ConvolutionalNeuralNetwork

def test_convolution_shape():
    """Test the convolution operation through its output layer shape

    """
    image_size = 64
    depth = 8
    strides = 2
    cnn = ConvolutionalNeuralNetwork("test", image_size)
    y = cnn.convolution(cnn.X, nb_filters=depth,
                        kernel_size=3, strides=strides, block_name="convtest")
    m = Model(cnn.X, y)
    output_shape = m.output_shape
    assert len(output_shape) == 4
    assert output_shape[1:] == (image_size//strides, image_size//strides, depth)

def test_transposed_convolution_shape():
    """Test the transposed convolution operation through its output layer shape

    """
    image_size = 64
    depth = 8
    kernel_size = 3
    strides = 2
    cnn = ConvolutionalNeuralNetwork("test", image_size)
    y = cnn.transposed_convolution(cnn.X, nb_filters=depth,
                                   kernel_size=kernel_size, strides=strides,
                                   block_name="transconvtest")
    m = Model(cnn.X, y)
    output_shape = m.output_shape
    assert len(output_shape) == 4
    assert output_shape[1:] == (image_size*strides, image_size*strides, depth)

def test_maxpooling_shape():
    """Test the max pooling operation through its output layer shape

    """
    image_size = 64
    nb_channels = 3
    psize = 2
    strides = 2
    cnn = ConvolutionalNeuralNetwork("test", image_size, nb_channels)
    y = cnn.maxpool(cnn.X, pool_size=psize, strides=strides, block_name="pooltest")
    m = Model(cnn.X, y)
    output_shape = m.output_shape
    assert len(output_shape) == 4
    assert output_shape[1:] == (image_size//strides, image_size//strides, nb_channels)

def test_dense_shape():
    """Test the fully-connected layer through its output shape

    """
    image_size = 64
    depth = 8
    cnn = ConvolutionalNeuralNetwork("test", image_size)
    y = cnn.dense(cnn.X, depth=depth, block_name="fctest")
    m = Model(cnn.X, y)
    output_shape = m.output_shape
    assert len(output_shape) == 4
    assert output_shape[1:] == (image_size, image_size, depth)

def test_flatten_shape():
    """Test the flattening layer through its output shape

    """
    image_size = 64
    nb_channels = 3
    cnn = ConvolutionalNeuralNetwork("test", image_size=image_size, nb_channels=nb_channels)
    y = cnn.flatten(cnn.X, block_name="flattentest")
    m = Model(cnn.X, y)
    output_shape = m.output_shape
    assert len(output_shape) == 2
    assert output_shape[1] == image_size * image_size * nb_channels

def test_layer_name():
    """Test the convolution operation through its output layer shape

    """
    image_size = 64
    depth = 8
    strides = 1
    cnn = ConvolutionalNeuralNetwork("test", image_size)
    y = cnn.convolution(cnn.X, nb_filters=depth,
                        kernel_size=3, strides=strides)
    y = cnn.convolution(y, nb_filters=depth,
                        kernel_size=3, strides=strides)
    m = Model(cnn.X, y)
    assert ([l.name for l in m.layers[1:]] ==
            ['conv2d_1', 'batch_normalization_1', 'activation_1',
             'conv2d_2', 'batch_normalization_2', 'activation_2'])
