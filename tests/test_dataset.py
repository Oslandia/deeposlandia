"""Unit test related to the dataset creation, population and loading
"""

import json
import os
import pytest

from deeposlandia.dataset import Dataset, MapillaryDataset, ShapeDataset

def test_dataset_creation(mapillary_image_size):
    """Create a generic dataset
    """
    d = Dataset(mapillary_image_size)
    assert d.image_size == mapillary_image_size
    assert d.get_nb_labels() == 0
    assert d.get_nb_images() == 0
    d.add_label(label_id=0, label_name="test_label", color=[100, 100, 100], is_evaluate=True)
    assert d.get_nb_labels() == 1

def test_mapillary_dataset_creation(mapillary_image_size, mapillary_nb_labels, mapillary_input_config):
    """Create a Mapillary dataset from a configuration file
    """
    d = MapillaryDataset(mapillary_image_size, mapillary_input_config)
    assert d.image_size == mapillary_image_size
    assert d.get_nb_labels(see_all=True) == mapillary_nb_labels
    assert d.get_nb_images() == 0

def test_mapillary_dataset_population(mapillary_image_size, mapillary_sample_dir,
                                      mapillary_nb_images, mapillary_nb_labels,
                                      mapillary_input_config, mapillary_config,
                                      mapillary_training_data):
    """Populate a Mapillary dataset
    """
    d = MapillaryDataset(mapillary_image_size, mapillary_input_config)
    d.populate(str(mapillary_training_data), mapillary_sample_dir, nb_images=mapillary_nb_images)
    d.save(str(mapillary_config))
    assert d.get_nb_labels(see_all=True) == mapillary_nb_labels
    assert d.get_nb_images() == mapillary_nb_images
    assert os.path.isfile(str(mapillary_config))
    assert all(len(os.listdir(os.path.join(str(mapillary_training_data), tmp_dir))) == mapillary_nb_images
               for tmp_dir in ["images", "labels"])

def test_mapillary_dataset_loading(mapillary_image_size, mapillary_nb_images,
                                   mapillary_input_config, mapillary_nb_labels,
                                   mapillary_config):
    """Load images into a Mapillary dataset
    """
    with open(mapillary_input_config) as fobj:
        config = json.load(fobj)
    d = MapillaryDataset(mapillary_image_size, mapillary_input_config)
    d.load(str(mapillary_config))
    assert d.get_nb_labels() == mapillary_nb_labels
    assert d.get_nb_images() == mapillary_nb_images

def test_shape_dataset_creation(shapes_image_size, shapes_nb_labels):
    """Create a Shapes dataset
    """
    d = ShapeDataset(shapes_image_size)
    assert d.image_size == shapes_image_size
    assert d.get_nb_labels() == shapes_nb_labels
    assert d.get_nb_images() == 0

def test_shape_dataset_population(shapes_image_size, shapes_nb_images, shapes_nb_labels,
                                  shapes_config, shapes_training_data):
    """Populate a Shapes dataset
    """
    d = ShapeDataset(shapes_image_size)
    d.populate(str(shapes_training_data), nb_images=shapes_nb_images)
    d.save(str(shapes_config))
    assert d.get_nb_labels() == shapes_nb_labels
    assert d.get_nb_images() == shapes_nb_images
    assert os.path.isfile(str(shapes_config))
    assert all(len(os.listdir(os.path.join(str(shapes_training_data), tmp_dir))) == shapes_nb_images
               for tmp_dir in ["images", "labels"])

def test_shape_dataset_loading(shapes_image_size, shapes_nb_images, shapes_nb_labels, shapes_config):
    """Load images into a Shapes dataset
    """
    d = ShapeDataset(shapes_image_size)
    d.load(str(shapes_config))
    assert d.get_nb_labels() == shapes_nb_labels
    assert d.get_nb_images() == shapes_nb_images
