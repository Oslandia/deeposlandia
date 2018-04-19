"""Unit test related to the generator building and feeding
"""

import pytest

import numpy as np

from deeposlandia import generator, utils


def test_feature_detection_labelling_concise():
    """Test `feature_detection_labelling` function in `generator` module by considering a concise
    labelling, *i.e.* all labels are represented into the array:
    * as a preliminary verification, check if passing string labels raises an AttributeError
    exception
    * test if output shape is first input shape (batch size) + an additional dimension given by the
    `label_ids` length
    * test if both representation provides the same information (native array on the first hand and
    its one-hot version on the second hand)
    """
    a = np.array([[[0, 0, 0, 2],
                   [3, 3, 0, 2],
                   [3, 3, 3, 0]],
                  [[2, 2, 0, 0],
                   [1, 2, 0, 0],
                   [2, 1, 0, 0]]])
    labels = np.unique(a).tolist()
    MIN, MAX = np.amin(a), np.amax(a)
    with pytest.raises(AttributeError):
        b = generator.feature_detection_labelling(a, ['0', '1', '2', '3'])
    b = generator.feature_detection_labelling(a, labels)
    assert b.shape == (a.shape[0], len(labels))
    assert b.tolist() == [[True, False, True, True],
                          [True, True, True, False]]


def test_feature_detection_labelling_sparse():
    """Test `feature_detection_labelling` function in `generator` module by considering a sparse
    labelling, *i.e.* the array contains unknown values (to mimic the non-evaluated label
    situations):
    * as a preliminary verification, check if passing string labels raises an AttributeError
    exception
    * test if label length is different from the list of values in the array
    * test if output shape is first input shape (batch size) + an additional dimension given by the
    `label_ids` length
    * test if both representation provides the same information (native array on the first hand and
    its one-hot version on the second hand)
    """
    a = np.array([[[0, 0, 0, 1, 1],
                   [3, 3, 0, 1, 1],
                   [3, 3, 3, 0, 0],
                   [3, 3, 3, 0, 0]],
                  [[1, 1, 2, 1, 2],
                   [3, 2, 2, 1, 3],
                   [1, 1, 1, 2, 1],
                   [1, 1, 2, 3, 2]]])
    labels = np.unique(a).tolist()[:-1]
    with pytest.raises(AttributeError):
        b = generator.feature_detection_labelling(a, ['0', '1', '2'])
    b = generator.feature_detection_labelling(a, labels)
    assert len(labels) != np.amax(a) - np.amin(a) + 1
    assert b.tolist() == [[True, True, False],
                          [False, True, True]]
    assert b.shape == (a.shape[0], len(labels))


def test_featdet_mapillary_generator():
    """Test the data generator for the Mapillary dataset
    """
    dataset = "mapillary"
    model = "feature_detection"
    IMAGE_SIZE = 128
    BATCH_SIZE = 10
    datapath = ("./tests/data/" + dataset + "/training")
    config = utils.read_config(datapath + ".json")
    label_ids = [x['id'] for x in config["labels"]]
    gen = generator.create_generator(dataset, model, datapath, IMAGE_SIZE, BATCH_SIZE, label_ids)
    item = next(gen)
    assert(len(item)==2)
    im_shape = item[0].shape
    assert im_shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
    label_shape = item[1].shape
    assert label_shape == (BATCH_SIZE, len(config['labels']))


def test_featdet_shape_generator():
    """Test the data generator for the shape dataset
    """
    dataset = "shapes"
    model = "feature_detection"
    IMAGE_SIZE = 48
    BATCH_SIZE = 10
    datapath = ("./tests/data/" + dataset + "/training")
    config = utils.read_config(datapath + ".json")
    label_ids = [x['id'] for x in config["labels"]]
    gen = generator.create_generator(dataset, model, datapath, IMAGE_SIZE, BATCH_SIZE, label_ids)
    item = next(gen)
    assert len(item) == 2
    im_shape = item[0].shape
    assert im_shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
    label_shape = item[1].shape
    assert label_shape == (BATCH_SIZE, len(config['labels']))


def test_semantic_segmentation_labelling_concise():
    """Test `semantic_segmentation_labelling` function in `generator` module by considering a
    concise labelling, *i.e.* the labels correspond to array values
    * as a preliminary verification, check if passing string labels raises an AttributeError
    exception
    * test if output shape is input shape + an additional dimension given by the
      `label_ids` length
    * test if both representation provides the same information (native array on the
      first hand and its one-hot version on the second hand)

    """
    a = np.array([[[1, 1, 3, 1],
                   [3, 3, 1, 1],
                   [3, 3, 3, 1]],
                  [[1, 1, 0, 0],
                   [2, 2, 0, 1],
                   [1, 1, 0, 0]]])
    labels = np.unique(a).tolist()
    asum, _ = np.histogram(a.reshape(-1), range=(np.amin(a), np.amax(a)))
    with pytest.raises(AttributeError):
        b = generator.semantic_segmentation_labelling(a, ['0', '1', '2', '3'])
    b = generator.semantic_segmentation_labelling(a, labels)
    assert b.shape == (a.shape[0], a.shape[1], a.shape[2], len(labels))
    assert b.tolist() == [[[[False, True, False, False],
                            [False, True, False, False],
                            [False, False, False, True],
                            [False, True, False, False]],
                           [[False, False, False, True],
                            [False, False, False, True],
                            [False, True, False, False],
                            [False, True, False, False]],
                           [[False, False, False, True],
                            [False, False, False, True],
                            [False, False, False, True],
                            [False, True, False, False]]],
                          [[[False, True, False, False],
                            [False, True, False, False],
                            [True, False, False, False],
                            [True, False, False, False]],
                           [[False, False, True, False],
                            [False, False, True, False],
                            [True, False, False, False],
                            [False, True, False, False]],
                           [[False, True, False, False],
                            [False, True, False, False],
                            [True, False, False, False],
                            [True, False, False, False]]]]


def test_semantic_segmentation_labelling_sparse():
    """Test `semantic_segmentation_labelling` function in `generator` module by considering a
    sparse labelling, *i.e.* the array contains unknown values (to mimic the non-evaluated label
    situations)
    * as a preliminary verification, check if passing string labels raises an AttributeError
    exception
    * test if output shape is input shape + an additional dimension given by the
      `label_ids` length
    * test if both representation provides the same information (native array on the
      first hand and its one-hot version on the second hand)

    """
    a = np.array([[[1, 1, 3, 1],
                   [3, 3, 1, 1],
                   [3, 4, 3, 1]],
                  [[1, 1, 0, 0],
                   [3, 4, 0, 1],
                   [1, 1, 0, 0]]])
    labels = [0, 2, 3]
    asum, _ = np.histogram(a.reshape(-1), range=(np.amin(a), np.amax(a)))
    with pytest.raises(AttributeError):
        b = generator.semantic_segmentation_labelling(a, ['0', '2', '3'])
    b = generator.semantic_segmentation_labelling(a, labels)
    assert len(labels) != np.amax(a) - np.amin(a) + 1
    assert b.shape == (a.shape[0], a.shape[1], a.shape[2], len(labels))
    assert b.tolist() == [[[[False, False, False],
                            [False, False, False],
                            [False, False, True],
                            [False, False, False]],
                           [[False, False, True],
                            [False, False, True],
                            [False, False, False],
                            [False, False, False]],
                           [[False, False, True],
                            [False, False, False],
                            [False, False, True],
                            [False, False, False]]],
                          [[[False, False, False],
                            [False, False, False],
                            [True, False, False],
                            [True, False, False]],
                           [[False, False, True],
                            [False, False, False],
                            [True, False, False],
                            [False, False, False]],
                           [[False, False, False],
                            [False, False, False],
                            [True, False, False],
                            [True, False, False]]]]


def test_semseg_mapillary_generator():
    """Test the data generator for the Mapillary dataset
    """
    dataset = "mapillary"
    model = "semantic_segmentation"
    IMAGE_SIZE = 128
    BATCH_SIZE = 10
    datapath = ("./tests/data/" + dataset + "/training")
    config = utils.read_config(datapath + ".json")
    label_ids = [x['id'] for x in config["labels"]]
    gen = generator.create_generator(dataset, model, datapath, IMAGE_SIZE, BATCH_SIZE, label_ids)
    item = next(gen)
    assert(len(item)==2)
    im_shape = item[0].shape
    assert im_shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
    label_shape = item[1].shape
    assert label_shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, len(config['labels']))


def test_semseg_shape_generator():
    """Test the data generator for the shape dataset
    """
    dataset = "shapes"
    model = "semantic_segmentation"
    IMAGE_SIZE = 48
    BATCH_SIZE = 10
    datapath = ("./tests/data/" + dataset + "/training")
    config = utils.read_config(datapath + ".json")
    label_ids = [x['id'] for x in config["labels"]]
    gen = generator.create_generator(dataset, model, datapath, IMAGE_SIZE, BATCH_SIZE, label_ids)
    item = next(gen)
    assert len(item) == 2
    im_shape = item[0].shape
    assert im_shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
    label_shape = item[1].shape
    assert label_shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, len(config['labels']))


def test_wrong_model_dataset_generator():
    """Test a wrong model and wrong dataset
    """
    dataset = "fake"
    model = "conquer_the_world"
    IMAGE_SIZE = 10
    BATCH_SIZE = 10
    datapath = ("./tests/data/" + dataset + "/training")
    label_ids = range(3)

    # wrong model name
    with pytest.raises(ValueError) as excinfo:
        generator.create_generator(dataset, 'feature_detection', datapath, IMAGE_SIZE, BATCH_SIZE, label_ids)
    assert str(excinfo.value) == "Wrong dataset name {}".format(dataset)

    # wrong model name
    with pytest.raises(ValueError) as excinfo:
        generator.create_generator('shapes', model, datapath, IMAGE_SIZE, BATCH_SIZE, label_ids)
    assert str(excinfo.value) == "Wrong model name {}".format(model)
