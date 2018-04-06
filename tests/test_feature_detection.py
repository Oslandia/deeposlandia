"""Unit test related to feature detection model instanciation
"""

from keras.models import Model

from deeposlandia.keras_feature_detection import FeatureDetectionNetwork

def test_simple_network_architecture():
    """Test the instanciation of a simple feature detection network

    """
    IMAGE_SIZE = 64
    NB_CHANNELS = 3
    NB_LABELS = 3
    net = FeatureDetectionNetwork("fd_test", image_size=IMAGE_SIZE,
                                  nb_channels=NB_CHANNELS, nb_labels=NB_LABELS)
    m = Model(net.X, net.Y)
    input_shape = m.input_shape
    output_shape = m.output_shape
    assert len(input_shape) == 4
    assert input_shape[1:] == (IMAGE_SIZE, IMAGE_SIZE, NB_CHANNELS)
    assert len(output_shape) == 2
    assert output_shape[1] == NB_LABELS

def test_vgg16_network_architecture():
    """Test the instanciation of a VGG16-inspired feature detection network

    """
    IMAGE_SIZE = 224
    NB_CHANNELS = 3
    NB_LABELS = 3
    net = FeatureDetectionNetwork("fd_test", image_size=IMAGE_SIZE,
                                  nb_channels=NB_CHANNELS, nb_labels=NB_LABELS,
                                  architecture="vgg16")
    m = Model(net.X, net.Y)
    input_shape = m.input_shape
    output_shape = m.output_shape
    assert len(input_shape) == 4
    assert input_shape[1:] == (IMAGE_SIZE, IMAGE_SIZE, NB_CHANNELS)
    assert len(output_shape) == 2
    assert output_shape[1] == NB_LABELS
