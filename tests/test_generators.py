"""Unit test related to the generator building and feeding
"""

import pytest

import numpy as np

from deeposlandia import generator, utils


def test_feature_detection_labelling_evaluated_labels():
    """Test `feature_detection_labelling` function by considering only evaluated labels, *i.e.* dataset
    labels that have a `is_evaluated` key at `False` are not integrated into the label generator

    """
    BATCH_SIZE = 5
    MIN = 0
    MAX = 10
    IMAGE_SIZE = 3
    a = np.random.randint(MIN, MAX, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE])
    labels = range(MIN, MAX)
    evaluated_labels = np.random.choice(range(MIN, MAX), MAX-3)
    b = generator.feature_detection_labelling(a, evaluated_labels)
    assert len(labels) != len(evaluated_labels)
    assert b.shape == (BATCH_SIZE, len(evaluated_labels))


def test_feature_detection_labelling():
    """Test `semantic_segmentation_labelling` function in `generator` module:
    - test if output shape is input shape + an additional dimension given by the `label_ids` length
    - test if both representation provides the same information (native array on the first hand and
    its one-hot version on the second hand)
    """
    BATCH_SIZE = 5
    MIN = 0
    MAX = 10
    IMAGE_SIZE = 3
    a = np.random.randint(MIN, MAX, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE])
    b = generator.feature_detection_labelling(a, range(MIN, MAX))
    assert b.shape == (BATCH_SIZE, MAX)


def test_feature_detection_labelling_wrong_label_id():
    """Test the value and the type of some label ids for feature detection one-hot encoding.
    """
    one_label = np.array([[0, 0, 0, 10, 10],
                          [3, 3, 0, 10, 10],
                          [3, 3, 3,  0,  0],
                          [3, 3, 3,  0,  0]])
    two_label = np.array([[10, 10, 0, 0, 0],
                          [0, 0, 0, 10, 10],
                          [10, 10, 0, 0, 0],
                          [10, 10, 0, 0, 0]])
    labels = np.array(one_label.tolist() + two_label.tolist())
    labels = labels.reshape((2, 4, 5, 1))
    # do not allow a list of string as label ids
    with pytest.raises(AttributeError):
        b = generator.feature_detection_labelling(labels, ['0', '3', '10'])

    # do not allow a label id value which is not in the 'labels' array
    with pytest.raises(AssertionError):
        b = generator.feature_detection_labelling(labels, [0, 2, 10])
    b = generator.feature_detection_labelling(labels, [0, 3, 10])
    assert b.tolist() ==[[True, True, True],
                         [True, False, True]]


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


def test_semantic_segmentation_labelling():
    """Test `semantic_segmentation_labelling` function in `generator` module

    - test if output shape is input shape + an additional dimension given by the
      `label_ids` length
    - test if both representation provides the same information (native array on the
      first hand and its one-hot version on the second hand)
    """
    MIN = 0
    MAX = 10
    SIZE = 3
    a = np.random.randint(MIN, MAX, [SIZE, SIZE])
    asum = np.bincount(a.reshape(-1))
    b = generator.semantic_segmentation_labelling(a, label_ids=range(MIN, MAX))
    bsum = np.sum(b, axis=(0, 1))
    assert b.shape == (SIZE, SIZE, MAX)
    assert list(bsum) == list(asum)


def test_semantic_segmentation_labelling_evaluated_labels():
    """Test `semantic_segmentation_labelling` function by considering only evaluated labels, *i.e.*
    dataset labels that have a `is_evaluated` key at `False` are not integrated into the label
    generator

    """
    BATCH_SIZE = 5
    MIN = 0
    MAX = 10
    IMAGE_SIZE = 3
    a = np.random.randint(MIN, MAX, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE])
    labels = range(MIN, MAX)
    evaluated_labels = np.random.choice(range(MIN, MAX), MAX-3)
    b = generator.semantic_segmentation_labelling(a, evaluated_labels)
    assert len(labels) != len(evaluated_labels)
    assert b.shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, len(evaluated_labels))



def test_semantic_segmentation_labelling_wrong_label_id():
    """Test if there are some AssertionError for some wrong label ids (type and value)
    """
    MIN = 0
    MAX = 10
    SIZE = 3
    a = np.random.randint(MIN, MAX, [SIZE, SIZE])

    with pytest.raises(AttributeError):
        b = generator.semantic_segmentation_labelling(a, ['0', '2', '10'])

    with pytest.raises(AssertionError):
        b = generator.semantic_segmentation_labelling(a, range(200, 210))


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
