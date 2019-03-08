"""Design a feature detection model with Keras API
"""

import daiquiri
import keras as K
from keras.applications import VGG16, inception_v3, resnet50

from deeposlandia.network import ConvolutionalNeuralNetwork


logger = daiquiri.getLogger(__name__)


class FeatureDetectionNetwork(ConvolutionalNeuralNetwork):
    """Class that encapsulates feature detection network creation

    Inherits from `ConvolutionalNeuralNetwork`

    Attributes
    ----------
    network_name : str
        Name of the network
    image_size : integer
        Input image size (height and width are equal)
    nb_channels : integer
        Number of input image channels (1 for greyscaled images, 3 for RGB
    images)
    nb_labels : integer
        Number of classes in the dataset glossary
    X : tensor
        (batch_size, image_size, image_size, nb_channels)-shaped input tensor;
    input image data
    Y : tensor
        (None, nb_classes)-shaped output tensor, built as the output of the
    last network layer
    """

    def __init__(
        self,
        network_name="mapillary",
        image_size=512,
        nb_channels=3,
        nb_labels=65,
        dropout=1.0,
        architecture="simple",
    ):
        super().__init__(
            network_name, image_size, nb_channels, nb_labels, dropout
        )
        if architecture == "vgg":
            self.Y = self.vgg16()
        elif architecture == "inception":
            self.Y = self.inception()
        elif architecture == "resnet":
            self.Y = self.resnet()
        elif architecture == "simple":
            self.Y = self.simple()
        else:
            logger.error(
                (
                    "Unknown network architecture. Please use "
                    "'simple', 'vgg', 'inception' or 'resnet'."
                )
            )
            raise ValueError("Unknown network architecture.")

    def output_layer(self, x, depth):
        """Build an output layer to a neural network, as a dense layer with
        sigmoid activation function as the point is to detect multiple labels
        on a single image

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
        y = K.layers.Dense(depth, name="output_fc")(x)
        y = K.layers.Activation(
            activation="sigmoid", name="output_activation"
        )(y)
        return y

    def simple(self):
        """Build a simple default convolutional neural network composed of 3
        convolution-maxpool blocks and 1 fully-connected layer

        Returns
        -------
        tensor
            (batch_size, nb_labels)-shaped output predictions, that have to be
        compared with ground-truth values
        """
        layer = self.convolution(
            self.X, nb_filters=16, kernel_size=7, block_name="conv1"
        )
        layer = self.maxpool(layer, pool_size=2, strides=2, block_name="pool1")
        layer = self.convolution(
            layer, nb_filters=32, kernel_size=5, block_name="conv2"
        )
        layer = self.maxpool(layer, pool_size=2, strides=2, block_name="pool2")
        layer = self.convolution(
            layer, nb_filters=64, kernel_size=3, block_name="conv3"
        )
        layer = self.maxpool(layer, pool_size=2, strides=2, block_name="pool3")
        layer = self.flatten(layer, block_name="flatten1")
        layer = self.dense(layer, depth=512, block_name="fc1")
        return self.output_layer(layer, depth=self.nb_labels)

    def vgg16(self):
        """Build the structure of a convolutional neural network from input
        image data to the last hidden layer on the model of a similar manner
        than VGG-net

        See: Simonyan & Zisserman, Very Deep Convolutional Networks for
        Large-Scale Image Recognition, arXiv technical report, 2014

        Returns
        -------
        tensor
            (batch_size, nb_labels)-shaped output predictions, that have to be
        compared with ground-truth values
        """
        vgg16_model = VGG16(input_tensor=self.X, include_top=False)
        y = self.flatten(vgg16_model.output, block_name="flatten")
        y = self.dense(y, 1024, block_name="fc1")
        y = self.dense(y, 1024, block_name="fc2")
        return self.output_layer(y, depth=self.nb_labels)

    def resnet(self):
        """Build the structure of a convolutional neural network from input
        image data to the last hidden layer on a similar manner than ResNet

        See: He, Zhang, Ren, Sun. Deep Residual Learning for Image
        Recognition. ArXiv technical report, 2015.

        Returns
        -------
        tensor
            (batch_size, nb_labels)-shaped output predictions, that have to be
        compared with ground-truth values
        """
        resnet_model = resnet50.ResNet50(
            include_top=False, input_tensor=self.X
        )
        y = self.flatten(resnet_model.output)
        return self.output_layer(y, depth=self.nb_labels)

    def inception(self):
        """Build the structure of a convolutional neural network from input
        image data to the last hidden layer on the model of a similar manner
        than Inception-V4

        See: Szegedy, Vanhoucke, Ioffe, Shlens. Rethinking the Inception
        Architecture for Computer Vision. ArXiv technical report, 2015.

        Returns
        -------
        tensor
            (batch_size, nb_labels)-shaped output predictions, that have to be
        compared with ground-truth values

        """
        inception_model = inception_v3.InceptionV3(
            input_tensor=self.X, include_top=False
        )
        y = K.layers.GlobalAveragePooling2D()(inception_model.output)
        return self.output_layer(y, depth=self.nb_labels)
