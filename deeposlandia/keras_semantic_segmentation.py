
"""Design a semantic segmentation model with Keras API
"""

import keras as K

from deeposlandia.network import ConvolutionalNeuralNetwork

class SemanticSegmentationNetwork(ConvolutionalNeuralNetwork):
    """
    """

    def __init__(self, network_name="mapillary", image_size=512, nb_channels=3,
                 nb_labels=65, learning_rate=1e-4, architecture="simple"):
        ConvolutionalNeuralNetwork.__init__(self, network_name, image_size,
                                            nb_channels, nb_labels, learning_rate)
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
            Output predictions, that have to be compared with ground-truth values
        """
        layer = self.convolution(self.X, nb_filters=32, kernel_size=3, name='conv1')
        layer = self.maxpool(layer, pool_size=2, strides=2, name='pool1')
        layer = self.convolution(layer, nb_filters=64, kernel_size=3, name='conv2')
        layer = self.maxpool(layer, pool_size=2, strides=2, name='pool2')
        layer = self.convolution(layer, nb_filters=128, kernel_size=3, name='conv3')
        layer = self.maxpool(layer, pool_size=2, strides=2, name='pool3')
        layer = self.transposed_convolution(layer, nb_filters=128, strides=2, kernel_size=3, name="trconv1")
        layer = self.transposed_convolution(layer, nb_filters=64, strides=2, kernel_size=3, name="trconv2")
        layer = self.transposed_convolution(layer, nb_filters=32, strides=2, kernel_size=3, name="trconv3")
        return self.output_layer(layer, depth=self._nb_labels)
