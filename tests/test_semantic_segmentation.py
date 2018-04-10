"""Unit test related to the semantic segmentation model
"""

from keras.models import Model

from deeposlandia.keras_semantic_segmentation import SemanticSegmentationNetwork

def test_network_instanciation():
    """Test a simple feature detection network

    """
    IMAGE_SIZE = 64
    NB_CHANNELS = 3
    NB_LABELS = 3
    net = SemanticSegmentationNetwork("ss_test", image_size=IMAGE_SIZE,
                                      nb_channels=NB_CHANNELS, nb_labels=NB_LABELS)
    m = Model(net.X, net.Y)
    input_shape = m.input_shape
    output_shape = m.output_shape
    assert len(input_shape) == 4
    assert input_shape[1:] == (IMAGE_SIZE, IMAGE_SIZE, NB_CHANNELS)
    assert len(output_shape) == 4
    assert output_shape[1:] == (IMAGE_SIZE, IMAGE_SIZE, NB_LABELS)
