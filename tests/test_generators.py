"""Unit test related to the generator building and feeding
"""

import numpy as np

from deeposlandia import generator, network, utils

def test_mapillary_generator():
    """Test the data generator for the Mapillary dataset

    """
    dataset = "mapillary"
    IMAGE_SIZE = 128
    BATCH_SIZE = 10
    datapath = ("./tests/data/" + dataset + "/training")
    config = utils.read_config("./tests/data/" + dataset + "/config_aggregate.json")
    gen = generator.create_generator(dataset, datapath, IMAGE_SIZE, BATCH_SIZE, config)
    item = next(gen)
    assert(len(item)==2)
    im_shape = item[0].shape
    assert im_shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
    label_shape = item[1].shape
    assert label_shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, len(config['labels']))

def test_shape_generator():
    """Test the data generator for the shape dataset

    """
    dataset = "shapes"
    IMAGE_SIZE = 48
    BATCH_SIZE = 10
    datapath = ("./tests/data/" + dataset + "/training")
    config = utils.read_config(datapath + ".json")
    gen = generator.create_generator(dataset, datapath, IMAGE_SIZE, BATCH_SIZE, config)
    item = next(gen)
    assert len(item) == 2
    im_shape = item[0].shape
    assert im_shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
    label_shape = item[1].shape
    assert label_shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, len(config['classes']))

def test_to_categorical():
    """Test `to_categorical` function in `generator` module:
    - test if output shape is input shape + an additional dimension given by the `label_ids` length
    - test if both representation provides the same information (native array on the first hand and
    its one-hot version on the second hand)

    """
    MIN = 0
    MAX = 10
    SIZE = 3
    a = np.random.randint(MIN, MAX, [SIZE, SIZE])
    asum = np.bincount(a.reshape(-1))
    b = generator.to_categorical(a, label_ids=range(MIN, MAX))
    bsum = np.sum(b, axis=(0, 1))
    assert b.shape == (SIZE, SIZE, MAX)
    assert list(bsum) == list(asum)
