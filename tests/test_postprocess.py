"""Unit tests dedicated to predicted label postprocessing
"""

import numpy as np

from deeposlandia import postprocess


def test_get_images():
    """Test the image getting function

    Preprocessed image filenames must end with ".png"
    """
    filenames = postprocess.get_images("./data", "tanzania", 384, "grid_034")
    assert np.all([f.endswith(".png") for f in filenames])


def test_extract_coordinates_from_filenames():
    """Test the x, y coordinates extraction from filenames

    The filename chosen convention is as follows:
        <path>/<name>_<width>_<height>_<x>_<y>.<extension>
    """
    filenames = ["./foofolder/foo_100_100_0_1000.png",
                 "./barfolder/bar_100_100_2000_2100.png"]
    coordinates = postprocess.extract_coordinates_from_filenames(filenames)
    assert np.all(coordinates == [[0, 1000], [2000, 2100]])


def test_assign_label_colors():
    """Test the label colourization function, that allows to replace label IDs
    with pixel triplets

    """
    labels = [{"name": "foo", "color": [0, 0, 0]},
              {"name": "bar", "color": [200, 200, 200]}]
    y = np.array([
        [[1, 1, 0], [1, 1, 0], [0, 1, 1]],
        [[0, 1, 1], [0, 0, 0], [0, 1, 1]]
    ])
    l1 = np.array([
        [[[200, 200, 200], [200, 200, 200], [0, 0, 0]],
         [[200, 200, 200], [200, 200, 200], [0, 0, 0]],
         [[0, 0, 0], [200, 200, 200], [200, 200, 200]]],
        [[[0, 0, 0], [200, 200, 200], [200, 200, 200]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [200, 200, 200], [200, 200, 200]]]
    ])
    l2 = postprocess.assign_label_colors(y, labels)
    assert np.all(l1 == l2)



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
    coordinates = [[x1, y1], [x2, y2],
                   [x3, y3], [x4, y4]]
    e1 = np.array(
        [[1, 2, 5, 6],
         [3, 4, 7, 8],
         [9, 10, 13, 14],
         [11, 12, 15, 16]],
        dtype=np.int8
    )
    e2 = postprocess.fill_labelled_image(tiles, coordinates, 2, 4)
    print(e1)
    print(e2)
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
    coordinates = [[x1, y1], [x2, y2],
                   [x3, y3], [x4, y4],
                   [x5, y5], [x6, y6]]
    e1 = np.array(
        [[1, 2, 5],
         [3, 4, 7],
         [9, 10, 13],
         [11, 12, 15],
         [17, 18, 21]],
        dtype=np.int8
    )
    e2 = postprocess.fill_labelled_image(tiles, coordinates, 2, 3, 5)
    print(e1)
    print(e2)
    assert np.all(e1 == e2)


def test_fill_labelled_image_different_sizes():
    """Test the numpy full labelled image building

    Work with label tables, hence fixtures must be (n, n)-shaped structures

    This test case corresponds to the situation where raw image width and height
    are not equal.
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
    coordinates = [[x1, y1], [x2, y2],
                   [x3, y3], [x4, y4],
                   [x5, y5], [x6, y6]]
    e1 = np.array(
        [[1, 2, 5, 6],
         [3, 4, 7, 8],
         [9, 10, 13, 14],
         [11, 12, 15, 16],
         [17, 18, 21, 22],
         [19, 20, 23, 24]],
        dtype=np.int8
    )
    e2 = postprocess.fill_labelled_image(tiles, coordinates, 2, 4, 6)
    assert np.all(e1 == e2)
