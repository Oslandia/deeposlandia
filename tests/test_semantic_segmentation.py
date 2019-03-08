"""Unit test related to the semantic segmentation model
"""

from keras.models import Model

from deeposlandia.semantic_segmentation import SemanticSegmentationNetwork


def test_network_instanciation(
    shapes_image_size, nb_channels, shapes_nb_labels
):
    """Test a simple feature detection network

    """
    net = SemanticSegmentationNetwork(
        "ss_test",
        image_size=shapes_image_size,
        nb_channels=nb_channels,
        nb_labels=shapes_nb_labels,
    )
    m = Model(net.X, net.Y)
    input_shape = m.input_shape
    output_shape = m.output_shape
    assert len(input_shape) == 4
    assert input_shape[1:] == (
        shapes_image_size,
        shapes_image_size,
        nb_channels,
    )
    assert len(output_shape) == 4
    assert output_shape[1:] == (
        shapes_image_size,
        shapes_image_size,
        shapes_nb_labels,
    )


def test_unet_network_architecture(
    mapillary_image_size, nb_channels, mapillary_nb_labels
):
    """Test a Unet architecture for semantic segmentation network

    """
    net = SemanticSegmentationNetwork(
        "ss_test",
        image_size=mapillary_image_size,
        nb_channels=nb_channels,
        nb_labels=mapillary_nb_labels,
    )
    m = Model(net.X, net.Y)
    input_shape = m.input_shape
    output_shape = m.output_shape
    assert len(input_shape) == 4
    assert input_shape[1:] == (
        mapillary_image_size,
        mapillary_image_size,
        nb_channels,
    )
    assert len(output_shape) == 4
    assert output_shape[1:] == (
        mapillary_image_size,
        mapillary_image_size,
        mapillary_nb_labels,
    )
