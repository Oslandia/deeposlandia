"""Unit tests dedicated to predicted label postprocessing
"""

import os

import numpy as np
import pytest

from deeposlandia import postprocess


def test_get_image_paths(tanzania_image_size):
    """Test the image path getting function

    Preprocessed image filenames must end with ".png"
    """
    filenames = postprocess.get_image_paths(
        f"./tests/data/tanzania/preprocessed/{tanzania_image_size}/testing/", "tanzania_sample"
    )
    assert np.all([f.endswith(".png") for f in filenames])


def test_extract_images(
    tanzania_image_size, tanzania_nb_output_testing_images
):
    """Test the image extraction function, that retrieve the accurate data in a
    'numpy.array' starting from a list of filenames

    This image data must be shaped as (nb_filenames, image_size, image_size,
    3).
    """
    filenames = postprocess.get_image_paths(
        f"./tests/data/tanzania/preprocessed/{tanzania_image_size}/testing/", "tanzania_sample"
    )
    images = postprocess.extract_images(filenames)
    assert len(images.shape) == 4
    assert images.shape[0] == tanzania_nb_output_testing_images
    assert images.shape[1] == tanzania_image_size
    assert images.shape[2] == tanzania_image_size
    assert images.shape[3] == 3


def test_get_labels(tanzania_image_size, tanzania_nb_labels):
    """Test the label retrieving from dataset glossary
    """
    labels = postprocess.get_labels(
        "./tests/data", "tanzania", tanzania_image_size
    )
    assert len(labels) == tanzania_nb_labels
    with pytest.raises(ValueError):
        postprocess.get_labels(
            "./elsewhere-on-the-disk", "wrong-dataset", tanzania_image_size
        )


def test_get_trained_model(tanzania_image_size, tanzania_nb_labels):
    """Test the semantic segmentation model retrieving

    Postprocessing implies getting a 'unet' model (i.e. semantic segmentation),
    hence the model must be shaped as follows:
    - an input layer of shape (None, image_size, image_size, 3)
    - an output layer of shape (None, image_size, image_size, nb_labels)
    - 69 hidden layers, amonst which:
      + 57 are related to convolution
      + 8 are related to upsampling
      + 4 are related to pooling
    """
    model = postprocess.get_trained_model(
        "./tests/data/tanzania/output/semseg/checkpoints/",
        tanzania_image_size,
        tanzania_nb_labels
    )
    assert model.input_shape[1:] == (
        tanzania_image_size,
        tanzania_image_size,
        3,
    )
    assert model.output_shape[1:] == (
        tanzania_image_size,
        tanzania_image_size,
        tanzania_nb_labels,
    )
    model_input_layer = [ml for ml in model.layers if "input" in ml.name]
    assert len(model_input_layer) == 1
    model_output_layer = [ml for ml in model.layers if "output" in ml.name]
    assert len(model_output_layer) == 1
    model_conv_layers = [ml for ml in model.layers if "conv" in ml.name]
    assert len(model_conv_layers) == 57
    model_up_layers = [ml for ml in model.layers if "up" in ml.name]
    assert len(model_up_layers) == 8
    model_pool_layers = [ml for ml in model.layers if "pool" in ml.name]
    assert len(model_pool_layers) == 4


def test_assign_label_colors():
    """Test the label colorization function, that allows to replace label IDs
    with pixel triplets
    """
    labels = [
        {"name": "foo", "color": [0, 0, 0]},
        {"name": "bar", "color": [200, 200, 200]},
    ]
    y = np.array(
        [[[1, 1, 0], [1, 1, 0], [0, 1, 1]], [[0, 1, 1], [0, 0, 0], [0, 1, 1]]]
    )
    expected_labels = np.array(
        [
            [
                [[200, 200, 200], [200, 200, 200], [0, 0, 0]],
                [[200, 200, 200], [200, 200, 200], [0, 0, 0]],
                [[0, 0, 0], [200, 200, 200], [200, 200, 200]],
            ],
            [
                [[0, 0, 0], [200, 200, 200], [200, 200, 200]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [200, 200, 200], [200, 200, 200]],
            ],
        ]
    )
    labels = postprocess.assign_label_colors(y, labels)
    assert np.all(labels == expected_labels)


def test_extract_coordinates_from_filenames():
    """Test the x, y coordinates extraction from filenames

    The filename chosen convention is as follows:
        <path>/<name>_<width>_<height>_<x>_<y>.<extension>
    """
    filenames = [
        "./foofolder/foo_100_100_0_1000.png",
        "./barfolder/bar_100_100_2000_2100.png",
    ]
    coordinates = postprocess.extract_coordinates_from_filenames(filenames)
    assert np.all(coordinates == [[0, 1000], [2000, 2100]])


def test_fill_labelled_image():
    """Test the numpy full labelled image building

    Work with pixel tables, hence fixtures must be (n, n, 3)-shaped structures
    """
    a1 = np.array([[1, 2], [3, 4]], dtype=np.int8)
    a2 = np.array([[5, 6], [7, 8]], dtype=np.int8)
    a3 = np.array([[9, 10], [11, 12]], dtype=np.int8)
    a4 = np.array([[13, 14], [15, 16]], dtype=np.int8)
    tiles = np.array([a1, a2, a3, a4])
    x1, y1 = 0, 0
    x2, y2 = 2, 0
    x3, y3 = 0, 2
    x4, y4 = 2, 2
    coordinates = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    e1 = np.array(
        [[1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14], [11, 12, 15, 16]],
        dtype=np.int8,
    )
    e2 = postprocess.fill_labelled_image(tiles, coordinates, 2, 4)
    assert np.all(e1 == e2)


def test_fill_labelled_image_incompatible_sizes():
    """Test the numpy full labelled image building

    Work with pixel tables, hence fixtures must be (n, n, 3)-shaped structures

    This test case corresponds to the situation where raw image dimensions are
    not multiples of the tile size.
    """
    a1 = np.array([[1, 2], [3, 4]], dtype=np.int8)
    a2 = np.array([[5, 6], [7, 8]], dtype=np.int8)
    a3 = np.array([[9, 10], [11, 12]], dtype=np.int8)
    a4 = np.array([[13, 14], [15, 16]], dtype=np.int8)
    a5 = np.array([[17, 18], [19, 20]], dtype=np.int8)
    a6 = np.array([[21, 22], [23, 24]], dtype=np.int8)
    tiles = np.array([a1, a2, a3, a4, a5, a6])
    x1, y1 = 0, 0
    x2, y2 = 2, 0
    x3, y3 = 0, 2
    x4, y4 = 2, 2
    x5, y5 = 0, 4
    x6, y6 = 2, 4
    coordinates = [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6]]
    e1 = np.array(
        [[1, 2, 5], [3, 4, 7], [9, 10, 13], [11, 12, 15], [17, 18, 21]],
        dtype=np.int8,
    )
    e2 = postprocess.fill_labelled_image(tiles, coordinates, 2, 3, 5)
    assert np.all(e1 == e2)


def test_fill_labelled_image_different_sizes():
    """Test the numpy full labelled image building

    Work with label tables, hence fixtures must be (n, n)-shaped structures

    This test case corresponds to the situation where raw image width and
    height are not equal.
    """
    a1 = np.array([[1, 2], [3, 4]], dtype=np.int8)
    a2 = np.array([[5, 6], [7, 8]], dtype=np.int8)
    a3 = np.array([[9, 10], [11, 12]], dtype=np.int8)
    a4 = np.array([[13, 14], [15, 16]], dtype=np.int8)
    a5 = np.array([[17, 18], [19, 20]], dtype=np.int8)
    a6 = np.array([[21, 22], [23, 24]], dtype=np.int8)
    tiles = np.array([a1, a2, a3, a4, a5, a6])
    x1, y1 = 0, 0
    x2, y2 = 2, 0
    x3, y3 = 0, 2
    x4, y4 = 2, 2
    x5, y5 = 0, 4
    x6, y6 = 2, 4
    coordinates = [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6]]
    e1 = np.array(
        [
            [1, 2, 5, 6],
            [3, 4, 7, 8],
            [9, 10, 13, 14],
            [11, 12, 15, 16],
            [17, 18, 21, 22],
            [19, 20, 23, 24],
        ],
        dtype=np.int8,
    )
    e2 = postprocess.fill_labelled_image(tiles, coordinates, 2, 4, 6)
    assert np.all(e1 == e2)


def test_build_full_labelled_image(
    tanzania_image_size, tanzania_nb_labels, tanzania_raw_image_size
):
    """Test the label prediction on a high-resolution image that has to be
        tiled during inference process

    The labelled output is composed of label IDs that must corresponds to the
    dataset glossary, and its shape must equal the original image size.
    """
    datapath = "./tests/data"
    dataset = "tanzania"
    image_paths = postprocess.get_image_paths(
        os.path.join(datapath, dataset, "preprocessed", str(tanzania_image_size), "testing"),
        "tanzania_sample"
    )
    images = postprocess.extract_images(image_paths)
    coordinates = postprocess.extract_coordinates_from_filenames(image_paths)
    model_filename = f"best-model-{tanzania_image_size}.h5"
    model = postprocess.get_trained_model(
        os.path.join(datapath, dataset, "output/semseg/checkpoints/", model_filename),
        tanzania_image_size,
        tanzania_nb_labels
    )
    labelled_image = postprocess.build_full_labelled_image(
        images,
        coordinates,
        model,
        tile_size=tanzania_image_size,
        img_width=tanzania_raw_image_size,
        batch_size=2,
    )
    assert labelled_image.shape == (
        tanzania_raw_image_size,
        tanzania_raw_image_size,
    )
    assert np.all(
        [ul in range(tanzania_nb_labels) for ul in np.unique(labelled_image)]
    )


def test_draw_grid(tanzania_image_size, tanzania_raw_image_size):
    """Test the grid drawing on a labelled image

    This function adds white (i.e. (255, 255, 255)) pixels each 'image_size'
    pixel vertically and horizontally, without modifying the input data shape.
    """
    data = np.zeros(
        [tanzania_raw_image_size, tanzania_raw_image_size, 3], dtype=np.uint8
    )
    gridded_data = postprocess.draw_grid(
        data,
        tanzania_raw_image_size,
        tanzania_raw_image_size,
        tanzania_image_size,
    )
    assert data.shape == gridded_data.shape
    expected_colors = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    colors = np.unique(np.reshape(gridded_data, [-1, 3]), axis=0)
    assert np.all(colors == expected_colors)
    expected_white_pixels = int(tanzania_raw_image_size / tanzania_image_size)
    gridded_line = gridded_data[0]
    white_horiz_pixels = np.all(gridded_line == [255, 255, 255], axis=1)
    assert sum(white_horiz_pixels) == expected_white_pixels
    gridded_column = gridded_data[:, 0]
    white_vertic_pixels = np.all(gridded_column == [255, 255, 255], axis=1)
    assert sum(white_vertic_pixels) == expected_white_pixels


def test_get_image_features(tanzania_example_image):
    """Test the image geographic feature recovering:
    - 'south', 'north', 'west' and 'east' are the image geographic coordinates,
    hence floating numbers
    - west is smaller than east
    - south is smaller than north
    - srid is an integer geocode
    - width and height are strictly positive int, as they represent the image
    size, in pixels
    """
    geofeatures = postprocess.get_image_features(
        "./tests/data", "tanzania", "tanzania_sample"
    )
    assert isinstance(geofeatures["south"], float)
    assert isinstance(geofeatures["north"], float)
    assert isinstance(geofeatures["east"], float)
    assert isinstance(geofeatures["west"], float)
    assert geofeatures["west"] < geofeatures["east"]
    assert geofeatures["south"] < geofeatures["north"]
    assert isinstance(geofeatures["srid"], int)
    assert isinstance(geofeatures["width"], int)
    assert isinstance(geofeatures["height"], int)
    assert geofeatures["width"] > 0
    assert geofeatures["height"] > 0
