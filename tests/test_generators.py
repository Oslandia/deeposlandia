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
    a = np.array([[[[10, 10, 200], [10, 10, 200], [10, 10, 200]],
                   [[200, 200, 200], [200, 200, 200], [10, 10, 200]],
                   [[200, 200, 200], [200, 200, 200], [200, 200, 200]]],
                  [[[10, 200, 10], [10, 200, 10], [10, 10, 200]],
                   [[200, 10, 10], [10, 200, 10], [10, 10, 200]],
                   [[10, 200, 10], [200, 10, 10], [10, 10, 200]]]])
    labels = np.unique(a.reshape(-1, 3), axis=0).tolist()
    wrong_config = [{'id': '0', 'color': [10, 10, 200], 'is_evaluate': True},
                    {'id': '1', 'color': [200, 10, 10], 'is_evaluate': True},
                    {'id': '2', 'color': [10, 200, 10], 'is_evaluate': True},
                    {'id': '3', 'color': [200, 200, 200], 'is_evaluate': True}]
    with pytest.raises(ValueError):
        b = generator.feature_detection_labelling(a, wrong_config)
    config = [{'id': 0, 'color': [10, 10, 200], 'is_evaluate': True},
              {'id': 1, 'color': [200, 10, 10], 'is_evaluate': True},
              {'id': 2, 'color': [10, 200, 10], 'is_evaluate': True},
              {'id': 3, 'color': [200, 200, 200], 'is_evaluate': True}]
    b = generator.feature_detection_labelling(a, config)
    assert b.shape == (a.shape[0], len(labels))
    assert b.tolist() == [[True, False, False, True],
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
    a = np.array([[[[10, 10, 200], [10, 10, 200], [10, 10, 200], [200, 10, 10]],
                   [[200, 200, 200], [200, 200, 200], [10, 10, 200], [200, 10, 10]],
                   [[200, 200, 200], [200, 200, 200], [200, 200, 200], [10, 10, 200]],
                   [[200, 200, 200], [200, 200, 200], [200, 200, 200], [10, 10, 200]]],
                  [[[200, 10, 10], [200, 10, 10], [10, 200, 10], [200, 10, 10]],
                   [[200, 200, 200], [10, 200, 10], [10, 200, 10], [10, 200, 10]],
                   [[200, 10, 10], [200, 10, 10], [200, 10, 10], [200, 200, 200]],
                   [[200, 10, 10], [200, 10, 10], [10, 200, 10], [200, 200, 200]]]])
    labels = np.unique(a.reshape(-1, 3), axis=0).tolist()[:-1]
    wrong_config = [{'id': '0', 'color': [10, 10, 200], 'is_evaluate': True},
                    {'id': '1', 'color': [200, 10, 10], 'is_evaluate': True},
                    {'id': '2', 'color': [10, 200, 10], 'is_evaluate': True}]
    with pytest.raises(ValueError):
        b = generator.feature_detection_labelling(a, wrong_config)
    config = [{'id': 0, 'color': [10, 10, 200], 'is_evaluate': True},
              {'id': 1, 'color': [200, 10, 10], 'is_evaluate': True},
              {'id': 2, 'color': [10, 200, 10], 'is_evaluate': True}]
    b = generator.feature_detection_labelling(a, config)
    assert len(labels) != np.amax(a) - np.amin(a) + 1
    assert b.tolist() == [[True, True, False],
                          [False, True, True]]
    assert b.shape == (a.shape[0], len(labels))


def test_featdet_mapillary_generator(mapillary_image_size,
                                     mapillary_sample,
                                     mapillary_sample_config,
                                     nb_channels):
    """Test the data generator for the Mapillary dataset
    """
    BATCH_SIZE = 10
    config = utils.read_config(mapillary_sample_config)
    label_ids = [x['id'] for x in config["labels"]]
    gen = generator.create_generator("mapillary", "feature_detection",
                                     mapillary_sample,
                                     mapillary_image_size,
                                     BATCH_SIZE,
                                     config["labels"])
    item = next(gen)
    assert(len(item)==2)
    im_shape = item[0].shape
    assert im_shape == (BATCH_SIZE, mapillary_image_size, mapillary_image_size, nb_channels)
    label_shape = item[1].shape
    assert label_shape == (BATCH_SIZE, len(label_ids))


def test_featdet_shape_generator(shapes_image_size, shapes_sample, shapes_sample_config, nb_channels):
    """Test the data generator for the shape dataset
    """
    BATCH_SIZE = 10
    config = utils.read_config(shapes_sample_config)
    label_ids = [x['id'] for x in config["labels"]]
    gen = generator.create_generator("shapes", "feature_detection", shapes_sample, shapes_image_size, BATCH_SIZE, config["labels"])
    item = next(gen)
    assert len(item) == 2
    im_shape = item[0].shape
    assert im_shape == (BATCH_SIZE, shapes_image_size, shapes_image_size, nb_channels)
    label_shape = item[1].shape
    assert label_shape == (BATCH_SIZE, len(label_ids))


def test_featdet_aerial_generator(aerial_image_size, aerial_sample, aerial_sample_config, nb_channels):
    """Test the data generator for the AerialImage dataset
    """
    BATCH_SIZE = 10
    config = utils.read_config(aerial_sample_config)
    label_ids = [x['id'] for x in config["labels"]]
    gen = generator.create_generator("aerial", "feature_detection", aerial_sample, aerial_image_size, BATCH_SIZE, config["labels"])
    item = next(gen)
    assert len(item) == 2
    im_shape = item[0].shape
    assert im_shape == (BATCH_SIZE, aerial_image_size, aerial_image_size, nb_channels)
    label_shape = item[1].shape
    assert label_shape == (BATCH_SIZE, len(label_ids))


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
    a = np.array([[[[200, 10, 10], [200, 10, 10], [200, 200, 200]],
                   [[200, 200, 200], [200, 200, 200], [200, 10, 10]],
                   [[200, 200, 200], [200, 200, 200], [200, 200, 200]]],
                  [[[200, 10, 10], [200, 10, 10], [10, 10, 200]],
                   [[10, 200, 10], [10, 200, 10], [10, 10, 200]],
                   [[200, 10, 10], [200, 10, 10], [10, 10, 200]]]])
    labels = np.unique(a.reshape(-1, 3), axis=0).tolist()
    wrong_config = [{'id': '0', 'color': [10, 10, 200], 'is_evaluate': True},
                    {'id': '1', 'color': [200, 10, 10], 'is_evaluate': True},
                    {'id': '2', 'color': [10, 200, 10], 'is_evaluate': True},
                    {'id': '3', 'color': [200, 200, 200], 'is_evaluate': True}]
    asum, _ = np.histogram(a.reshape(-1), range=(np.amin(a), np.amax(a)))
    with pytest.raises(ValueError):
        b = generator.semantic_segmentation_labelling(a, wrong_config)
    config = [{'id': 0, 'color': [10, 10, 200], 'is_evaluate': True},
              {'id': 1, 'color': [200, 10, 10], 'is_evaluate': True},
              {'id': 2, 'color': [10, 200, 10], 'is_evaluate': True},
              {'id': 3, 'color': [200, 200, 200], 'is_evaluate': True}]
    b = generator.semantic_segmentation_labelling(a, config)
    assert b.shape == (a.shape[0], a.shape[1], a.shape[2], len(labels))
    assert b.tolist() == [[[[False, True, False, False],
                            [False, True, False, False],
                            [False, False, False, True]],
                           [[False, False, False, True],
                            [False, False, False, True],
                            [False, True, False, False]],
                           [[False, False, False, True],
                            [False, False, False, True],
                            [False, False, False, True]]],
                          [[[False, True, False, False],
                            [False, True, False, False],
                            [True, False, False, False]],
                           [[False, False, True, False],
                            [False, False, True, False],
                            [True, False, False, False]],
                           [[False, True, False, False],
                            [False, True, False, False],
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
    a = np.array([[[[200, 10, 10], [200, 10, 10], [200, 200, 200]],
                   [[200, 200, 200], [200, 200, 200], [200, 10, 10]],
                   [[200, 200, 200], [100, 100, 100], [200, 200, 200]]],
                  [[[200, 10, 10], [200, 10, 10], [10, 10, 200]],
                   [[200, 200, 200], [100, 100, 100], [10, 10, 200]],
                   [[200, 10, 10], [200, 10, 10], [10, 10, 200]]]])
    asum, _ = np.histogram(a.reshape(-1), range=(np.amin(a), np.amax(a)))
    wrong_config = [{'id': '0', 'color': [10, 10, 200], 'is_evaluate': True},
                    {'id': '2', 'color': [10, 200, 10], 'is_evaluate': True},
                    {'id': '3', 'color': [200, 200, 200], 'is_evaluate': True}]
    with pytest.raises(ValueError):
        b = generator.semantic_segmentation_labelling(a, wrong_config)
    config = [{'id': 0, 'color': [10, 10, 200], 'is_evaluate': True},
              {'id': 2, 'color': [10, 200, 10], 'is_evaluate': True},
              {'id': 3, 'color': [200, 200, 200], 'is_evaluate': True}]
    labels = [item["id"] for item in config]
    b = generator.semantic_segmentation_labelling(a, config)
    assert len(labels) != np.amax(a) - np.amin(a) + 1
    assert b.shape == (a.shape[0], a.shape[1], a.shape[2], len(labels))
    assert b.tolist() == [[[[False, False, False],
                            [False, False, False],
                            [False, False, True]],
                           [[False, False, True],
                            [False, False, True],
                            [False, False, False]],
                           [[False, False, True],
                            [False, False, False],
                            [False, False, True]]],
                          [[[False, False, False],
                            [False, False, False],
                            [True, False, False]],
                           [[False, False, True],
                            [False, False, False],
                            [True, False, False]],
                           [[False, False, False],
                            [False, False, False],
                            [True, False, False]]]]


def test_semseg_mapillary_generator(mapillary_image_size,
                                    mapillary_sample,
                                    mapillary_sample_config,
                                    nb_channels):
    """Test the data generator for the Mapillary dataset
    """
    BATCH_SIZE = 10
    config = utils.read_config(mapillary_sample_config)
    label_ids = [x['id'] for x in config["labels"]]
    gen = generator.create_generator("mapillary", "semantic_segmentation",
                                     mapillary_sample,
                                     mapillary_image_size,
                                     BATCH_SIZE, config["labels"])
    item = next(gen)
    assert(len(item)==2)
    im_shape = item[0].shape
    assert im_shape == (BATCH_SIZE, mapillary_image_size, mapillary_image_size, nb_channels)
    label_shape = item[1].shape
    assert label_shape == (BATCH_SIZE, mapillary_image_size, mapillary_image_size, len(label_ids))


def test_semseg_shape_generator(shapes_image_size, shapes_sample, shapes_sample_config, nb_channels):
    """Test the data generator for the shape dataset
    """
    BATCH_SIZE = 10
    config = utils.read_config(shapes_sample_config)
    label_ids = [x['id'] for x in config["labels"]]
    gen = generator.create_generator("shapes", "semantic_segmentation",
                                     shapes_sample, shapes_image_size,
                                     BATCH_SIZE, config["labels"])
    item = next(gen)
    assert len(item) == 2
    im_shape = item[0].shape
    assert im_shape == (BATCH_SIZE, shapes_image_size, shapes_image_size, nb_channels)
    label_shape = item[1].shape
    assert label_shape == (BATCH_SIZE, shapes_image_size, shapes_image_size, len(label_ids))


def test_semseg_aerial_generator(aerial_image_size, aerial_sample,
                                 aerial_sample_config, nb_channels):
    """Test the data generator for the AerialImage dataset
    """
    BATCH_SIZE = 10
    config = utils.read_config(aerial_sample_config)
    label_ids = [x['id'] for x in config["labels"]]
    gen = generator.create_generator("aerial", "semantic_segmentation",
                                     aerial_sample,
                                     aerial_image_size,
                                     BATCH_SIZE, config["labels"])
    item = next(gen)
    assert(len(item)==2)
    im_shape = item[0].shape
    assert im_shape == (BATCH_SIZE, aerial_image_size, aerial_image_size, nb_channels)
    label_shape = item[1].shape
    assert label_shape == (BATCH_SIZE, aerial_image_size, aerial_image_size, len(label_ids))


def test_wrong_model_dataset_generator(shapes_sample_config):
    """Test a wrong model and wrong dataset
    """
    dataset = "fake"
    model = "conquer_the_world"
    IMAGE_SIZE = 10
    BATCH_SIZE = 10
    datapath = ("./tests/data/" + dataset + "/training")
    config = utils.read_config(shapes_sample_config)

    # wrong dataset name
    with pytest.raises(ValueError) as excinfo:
        generator.create_generator(dataset, 'feature_detection', datapath, IMAGE_SIZE, BATCH_SIZE, config["labels"])
    assert str(excinfo.value) == "Wrong dataset name {}".format(dataset)

    # wrong model name
    with pytest.raises(ValueError) as excinfo:
        generator.create_generator('shapes', model, datapath, IMAGE_SIZE, BATCH_SIZE, config["labels"])
    assert str(excinfo.value) == "Wrong model name {}".format(model)
