"""Design a semantic segmentation model with Keras API
"""

import daiquiri
import keras as K

from deeposlandia.network import ConvolutionalNeuralNetwork


logger = daiquiri.getLogger(__name__)


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
        Number of input image channels (1 for greyscaled images, 3 for RGB
    images)
    nb_labels : integer
        Number of classes in the dataset glossary
    X : tensor
        (batch_size, image_size, image_size, nb_channels)-shaped input tensor;
    input image data
    Y : tensor
        (batch_size, image_size, image_size, nb_classes)-shaped output tensor,
    built as the output of the last network layer
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
        if architecture == "unet":
            error_msg = (
                "Please consider a divisible-per-16 image size "
                "for this architecture."
            )
            assert image_size % 16 == 0, error_msg
            self.Y = self.unet()
        elif architecture == "dilated":
            self.Y = self.dilated(add_context=True)
        elif architecture == "simple":
            self.Y = self.simple()
        else:
            logger.error(
                (
                    "Unknown network architecture. Please use "
                    "'simple' or 'unet'."
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
        y = K.layers.Conv2DTranspose(
            depth, kernel_size=2, padding="same", name="output_trconv"
        )(x)
        y = K.layers.Activation("softmax", name="output_activation")(y)
        return y

    def simple(self):
        """Build a simple default convolutional neural network composed of 3
        convolution-maxpool blocks and 1 fully-connected layer

        Returns
        -------
        tensor
            (batch_size, image_size, image_size, nb_labels)-shaped output
        predictions, that have to be compared with ground-truth values
        """
        layer = self.convolution(
            self.X, nb_filters=32, kernel_size=3, block_name="conv1"
        )
        layer = self.maxpool(layer, pool_size=2, strides=2, block_name="pool1")
        layer = self.convolution(
            layer, nb_filters=64, kernel_size=3, block_name="conv2"
        )
        layer = self.maxpool(layer, pool_size=2, strides=2, block_name="pool2")
        layer = self.convolution(
            layer, nb_filters=128, kernel_size=3, block_name="conv3"
        )
        layer = self.maxpool(layer, pool_size=2, strides=2, block_name="pool3")
        layer = self.transposed_convolution(
            layer,
            nb_filters=128,
            strides=2,
            kernel_size=3,
            block_name="trconv1",
        )
        layer = self.transposed_convolution(
            layer,
            nb_filters=64,
            strides=2,
            kernel_size=3,
            block_name="trconv2",
        )
        layer = self.transposed_convolution(
            layer,
            nb_filters=32,
            strides=2,
            kernel_size=3,
            block_name="trconv3",
        )
        return self.output_layer(layer, depth=self.nb_labels)

    def unet(self):
        """Build a U-net convolutional neural network; this architecture is
        characterized by a first set of convolution layer groups, each of which
        being followed by a max pooling layer so as to decrease the image size,
        and by a second set of convolution layer groups that are sprinkled by
        upsampling layers. A link between the first decreasing part and the
        second increasing part is ensured by crosswise concatenation
        operations.

        See: Ronneberger, Fischer & Brox, U-Net: Convolutional Networks for
        Biomedical Image Segmentation, arXiv technical report, 2015

        Returns
        -------
        tensor
            (batch_size, image_size, image_size, nb_labels)-shaped output
        predictions, that have to be compared with ground-truth values

        """
        conv1 = self.convolution(
            self.X, nb_filters=32, kernel_size=3, block_name="conv1a"
        )
        conv1 = self.convolution(
            conv1, nb_filters=32, kernel_size=3, block_name="conv1b"
        )

        pool1 = self.maxpool(conv1, pool_size=2, strides=2, block_name="pool1")
        conv2 = self.convolution(
            pool1, nb_filters=64, kernel_size=3, block_name="conv2a"
        )
        conv2 = self.convolution(
            conv2, nb_filters=64, kernel_size=3, block_name="conv2b"
        )

        pool2 = self.maxpool(conv2, pool_size=2, strides=2, block_name="pool2")
        conv3 = self.convolution(
            pool2, nb_filters=128, kernel_size=3, block_name="conv3a"
        )
        conv3 = self.convolution(
            conv3, nb_filters=128, kernel_size=3, block_name="conv3b"
        )

        pool3 = self.maxpool(conv3, pool_size=2, strides=2, block_name="pool3")
        conv4 = self.convolution(
            pool3, nb_filters=256, kernel_size=3, block_name="conv4a"
        )
        conv4 = self.convolution(
            conv4, nb_filters=256, kernel_size=3, block_name="conv4b"
        )

        pool4 = self.maxpool(conv4, pool_size=2, strides=2, block_name="pool4")
        conv5 = self.convolution(
            pool4, nb_filters=512, kernel_size=3, block_name="conv5a"
        )
        conv5 = self.convolution(
            conv5, nb_filters=512, kernel_size=3, block_name="conv5b"
        )

        up1 = self.upsample(conv5, conv4, block_name="up1")
        conv6 = self.convolution(
            up1, nb_filters=256, kernel_size=3, block_name="conv6a"
        )
        conv6 = self.convolution(
            conv6, nb_filters=256, kernel_size=3, block_name="conv6b"
        )

        up2 = self.upsample(conv6, conv3, block_name="up2")
        conv7 = self.convolution(
            up2, nb_filters=128, kernel_size=3, block_name="conv7a"
        )
        conv7 = self.convolution(
            conv7, nb_filters=128, kernel_size=3, block_name="conv7b"
        )

        up3 = self.upsample(conv7, conv2, block_name="up3")
        conv8 = self.convolution(
            up3, nb_filters=64, kernel_size=3, block_name="conv8a"
        )
        conv8 = self.convolution(
            conv8, nb_filters=64, kernel_size=3, block_name="conv8b"
        )

        up4 = self.upsample(conv8, conv1, block_name="up4")
        conv9 = self.convolution(
            up4, nb_filters=32, kernel_size=3, block_name="conv9a"
        )
        conv9 = self.convolution(
            conv9, nb_filters=32, kernel_size=3, block_name="conv9b"
        )
        conv9 = self.convolution(
            conv9,
            nb_filters=self.nb_labels,
            kernel_size=1,
            block_name="conv9c",
        )
        return K.layers.Activation("softmax", name="output_activation")(conv9)

    def dilated(self, add_context=False):
        """Build a dilated convolution network with two parts: a front-end part
        which is essentially an adapted VGG network, and a context part that
        corresponds to a sequence of layers with dilated convolutions.

        See: Yu, Koltun (2016). Multi-scale context aggregation by dilated
        convolutions, ArXiV technical report. Available at
        https://arxiv.org/pdf/1511.07122.pdf

        Parameters
        ----------
        add_context : boolean
             If true, the context block is added to the architecture

        Returns
        -------
        tensor
            (batch_size, image_size, image_size, nb_labels)-shaped output
        predictions, that have to be compared with ground-truth values

        """
        conv1 = self.convolution(
            self.X,
            nb_filters=64,
            kernel_size=3,
            batch_norm=False,
            block_name="conv1a_fe",
        )
        conv1 = self.convolution(
            conv1,
            nb_filters=64,
            kernel_size=3,
            batch_norm=False,
            block_name="conv1b_fe",
        )
        pool1 = self.maxpool(
            conv1, pool_size=2, strides=2, block_name="pool1_fe"
        )
        conv2 = self.convolution(
            pool1,
            nb_filters=128,
            kernel_size=3,
            batch_norm=False,
            block_name="conv2a_fe",
        )
        conv2 = self.convolution(
            conv2,
            nb_filters=128,
            kernel_size=3,
            batch_norm=False,
            block_name="conv2b_fe",
        )
        pool2 = self.maxpool(
            conv2, pool_size=2, strides=2, block_name="pool2_fe"
        )

        conv3 = self.convolution(
            pool2,
            nb_filters=256,
            kernel_size=3,
            batch_norm=False,
            block_name="conv3a_fe",
        )
        conv3 = self.convolution(
            conv3,
            nb_filters=256,
            kernel_size=3,
            batch_norm=False,
            block_name="conv3b_fe",
        )
        conv3 = self.convolution(
            conv3,
            nb_filters=256,
            kernel_size=3,
            batch_norm=False,
            block_name="conv3c_fe",
        )
        pool3 = self.maxpool(
            conv3, pool_size=2, strides=2, block_name="pool3_fe"
        )

        conv4 = self.convolution(
            pool3,
            nb_filters=512,
            kernel_size=3,
            batch_norm=False,
            block_name="conv4a_fe",
        )
        conv4 = self.convolution(
            conv4,
            nb_filters=512,
            kernel_size=3,
            batch_norm=False,
            block_name="conv4b_fe",
        )
        conv4 = self.convolution(
            conv4,
            nb_filters=512,
            kernel_size=3,
            batch_norm=False,
            block_name="conv4c_fe",
        )

        conv5 = self.convolution(
            conv4,
            nb_filters=512,
            kernel_size=3,
            batch_norm=False,
            dilation_rate=2,
            block_name="conv5a_fe",
        )
        conv5 = self.convolution(
            conv5,
            nb_filters=512,
            kernel_size=3,
            batch_norm=False,
            dilation_rate=2,
            block_name="conv5b_fe",
        )
        conv5 = self.convolution(
            conv5,
            nb_filters=512,
            kernel_size=3,
            batch_norm=False,
            dilation_rate=2,
            block_name="conv5c_fe",
        )

        conv6 = self.convolution(
            conv5,
            nb_filters=4096,
            kernel_size=7,
            batch_norm=False,
            dilation_rate=4,
            block_name="conv6_fe",
        )
        conv6 = K.layers.Dropout(0.5, name="do1")(conv6)

        conv7 = self.convolution(
            conv6,
            nb_filters=4096,
            kernel_size=1,
            batch_norm=False,
            block_name="conv7_fe",
        )
        conv7 = K.layers.Dropout(0.5, name="do2")(conv7)

        conv8 = self.convolution(
            conv7,
            nb_filters=self.nb_labels,
            batch_norm=False,
            kernel_size=1,
            activation="linear",
            block_name="conv8_fe",
        )

        if add_context:
            dilated_net = self.add_dilated_context(conv8)
        else:
            dilated_net = conv8

        # upsample
        net = self.transposed_convolution(
            dilated_net,
            nb_filters=256,
            batch_norm=False,
            kernel_size=3,
            strides=2,
            block_name="trconv1",
        )
        net = self.transposed_convolution(
            net,
            nb_filters=256,
            batch_norm=False,
            kernel_size=3,
            strides=2,
            block_name="trconv2",
        )
        net = self.transposed_convolution(
            net,
            nb_filters=self.nb_labels,
            batch_norm=False,
            activation="softmax",
            kernel_size=3,
            strides=2,
            block_name="trconv3",
        )
        return net
