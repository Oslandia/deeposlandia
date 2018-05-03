
"""Design a semantic segmentation model with Keras API
"""

import keras as K

from deeposlandia.network import ConvolutionalNeuralNetwork

class SemanticSegmentationNetwork(ConvolutionalNeuralNetwork):
    """Class that encapsulates semantic segmentation network creation

    Inherits from `ConvolutionalNeuralNetwork`

    Attributes
    ----------
    network_name : str
        Name of the network
    image_size : integer
        Input image size (height and width are equal)
    nb_channels : integer
        Number of input image channels (1 for greyscaled images, 3 for RGB images)
    nb_labels : integer
        Number of classes in the dataset glossary
    X : tensor
        (batch_size, image_size, image_size, nb_channels)-shaped input tensor; input image data
    Y : tensor
        (batch_size, image_size, image_size, nb_classes)-shaped output tensor, built as the output of the
    last network layer
    """

    def __init__(self, network_name="mapillary", image_size=512, nb_channels=3,
                 nb_labels=65, dropout=1.0, architecture="simple"):
        super().__init__(network_name, image_size, nb_channels, nb_labels, dropout)
        self.Y = self.simple()

    def output_layer(self, x, depth):
        """Build an output layer to a neural network, as a dense layer with sigmoid activation
        function as the point is to detect multiple labels on a single image

        Parameters
        ----------
        x : tensor
            Previous layer within the neural network (last hidden layer)
        depth : integer
            Dimension of the previous neural network layer

        Returns
        -------
        tensor
            2D output layer
        """
        y = K.layers.Conv2DTranspose(depth, kernel_size=2, padding="same", name='output_trconv')(x)
        y = K.layers.Activation('softmax', name='output_activation')(y)
        return y

    def simple(self):
        """Build a simple default convolutional neural network composed of 3 convolution-maxpool
        blocks and 1 fully-connected layer

        Returns
        -------
        tensor
            (batch_size, image_size, image_size, nb_labels)-shaped output predictions, that have to
        be compared with ground-truth values
        """
        layer = self.convolution(self.X, nb_filters=32, kernel_size=3, block_name='conv1')
        layer = self.maxpool(layer, pool_size=2, strides=2, block_name='pool1')
        layer = self.convolution(layer, nb_filters=64, kernel_size=3, block_name='conv2')
        layer = self.maxpool(layer, pool_size=2, strides=2, block_name='pool2')
        layer = self.convolution(layer, nb_filters=128, kernel_size=3, block_name='conv3')
        layer = self.maxpool(layer, pool_size=2, strides=2, block_name='pool3')
        layer = self.transposed_convolution(layer, nb_filters=128, strides=2, kernel_size=3, block_name="trconv1")
        layer = self.transposed_convolution(layer, nb_filters=64, strides=2, kernel_size=3, block_name="trconv2")
        layer = self.transposed_convolution(layer, nb_filters=32, strides=2, kernel_size=3, block_name="trconv3")
        return self.output_layer(layer, depth=self.nb_labels)
