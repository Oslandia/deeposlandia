"""Unit test associated to the generator building and feeding
"""

import numpy as np
import keras as K

from deeposlandia import generator, network, utils

def test_mapillary_generator():
    """Test the data generator for the Mapillary dataset

    """
    dataset = "mapillary"
    IMAGE_SIZE = 256
    BATCH_SIZE = 10
    datapath = ("data/mapillary/preprocessed/" + str(IMAGE_SIZE)
                + "_aggregated/training/")
    config = utils.read_config("./data/mapillary/input/config_aggregate.json")
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
    IMAGE_SIZE = 64
    BATCH_SIZE = 10
    datapath = ("data/shapes/preprocessed/" + str(IMAGE_SIZE)
                + "_full/training")
    config = utils.read_config(datapath + ".json")
    gen = generator.create_generator(dataset, datapath, IMAGE_SIZE, BATCH_SIZE, config)
    item = next(gen)
    assert len(item) == 2
    im_shape = item[0].shape
    assert im_shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
    label_shape = item[1].shape
    assert label_shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, len(config['classes']))
