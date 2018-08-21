"""Unit test related to the dataset creation, population and loading
"""

import json
import os
import pytest

from deeposlandia.dataset import (Dataset, MapillaryDataset,
                                  ShapeDataset, AerialDataset)

def test_dataset_creation(mapillary_image_size):
    """Create a generic dataset
    """
    d = Dataset(mapillary_image_size)
    assert d.image_size == mapillary_image_size
    assert d.get_nb_labels() == 0
    assert d.get_nb_images() == 0
    d.add_label(label_id=0, label_name="test_label", color=[100, 100, 100], is_evaluate=True)
    assert d.get_nb_labels() == 1

def test_mapillary_dataset_creation(mapillary_image_size, mapillary_nb_labels,
    mapillary_input_config):
    """Create a Mapillary dataset from a configuration file
    """
    d = MapillaryDataset(mapillary_image_size, mapillary_input_config)
    assert d.image_size == mapillary_image_size
    assert d.get_nb_labels(see_all=True) == mapillary_nb_labels
    assert d.get_nb_images() == 0

def test_mapillary_dataset_population(mapillary_image_size,
                                      mapillary_raw_sample,
                                      mapillary_nb_images, mapillary_nb_labels,
                                      mapillary_input_config, mapillary_config,
                                      mapillary_temp_dir):
    """Populate a Mapillary dataset
    """
    d = MapillaryDataset(mapillary_image_size, mapillary_input_config)
    d.populate(str(mapillary_temp_dir), mapillary_raw_sample, nb_images=mapillary_nb_images)
    d.save(str(mapillary_config))
    assert d.get_nb_labels(see_all=True) == mapillary_nb_labels
    assert d.get_nb_images() == mapillary_nb_images
    assert os.path.isfile(str(mapillary_config))
    assert all(len(os.listdir(os.path.join(str(mapillary_temp_dir), tmp_dir))) == mapillary_nb_images
               for tmp_dir in ["images", "labels"])

def test_mapillary_dataset_population_without_labels(mapillary_image_size, mapillary_input_config,
                                                     mapillary_sample_without_labels_dir,
                                                     mapillary_nb_images, mapillary_temp_dir):
    """Fail at populating a Mapillary dataset without labelled images
    """
    d = MapillaryDataset(mapillary_image_size, mapillary_input_config)
    with pytest.raises(FileNotFoundError) as excinfo:
        d.populate(str(mapillary_temp_dir), mapillary_sample_without_labels_dir, nb_images=mapillary_nb_images)
    assert str(excinfo.value).split(':')[0] == "[Errno 2] No such file or directory"

def test_mapillary_dataset_loading(mapillary_image_size, mapillary_nb_images,
                                   mapillary_input_config, mapillary_nb_labels,
                                   mapillary_sample_config):
    """Load images into a Mapillary dataset
    """
    with open(mapillary_input_config) as fobj:
        config = json.load(fobj)
    d = MapillaryDataset(mapillary_image_size, mapillary_input_config)
    d.load(mapillary_sample_config)
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
                                  shapes_config, shapes_temp_dir):
    """Populate a Shapes dataset
    """
    d = ShapeDataset(shapes_image_size)
    d.populate(str(shapes_temp_dir), nb_images=shapes_nb_images)
    d.save(str(shapes_config))
    assert d.get_nb_labels() == shapes_nb_labels
    assert d.get_nb_images() == shapes_nb_images
    assert os.path.isfile(str(shapes_config))
    assert all(len(os.listdir(os.path.join(str(shapes_temp_dir), tmp_dir))) == shapes_nb_images
               for tmp_dir in ["images", "labels"])

def test_shape_dataset_loading(shapes_image_size, shapes_nb_images, shapes_nb_labels, shapes_sample_config):
    """Load images into a Shapes dataset
    """
    d = ShapeDataset(shapes_image_size)
    d.load(shapes_sample_config)
    assert d.get_nb_labels() == shapes_nb_labels
    assert d.get_nb_images() == shapes_nb_images


def test_aerial_dataset_creation(aerial_image_size, aerial_tile_size,
                                 aerial_nb_labels):
    """Create a AerialImage dataset
    """
    d = AerialDataset(aerial_tile_size)
    assert d.image_size == aerial_image_size
    assert d.get_nb_labels() == aerial_nb_labels
    assert d.get_nb_images() == 0

def test_aerial_dataset_population(aerial_tile_size, aerial_temp_dir,
                                   aerial_raw_sample, aerial_nb_images,
                                   aerial_config, aerial_nb_labels,
                                   aerial_nb_output_images):
    """Populate a AerialImage dataset
    """
    d = AerialDataset(aerial_tile_size)
    d.populate(str(aerial_temp_dir), aerial_raw_sample,
               nb_images=aerial_nb_images)
    d.save(str(aerial_config))
    assert d.get_nb_labels() == aerial_nb_labels
    assert d.get_nb_images() == aerial_nb_output_images
    assert os.path.isfile(str(aerial_config))
    assert all(len(os.listdir(os.path.join(str(aerial_temp_dir), tmp_dir))) == aerial_nb_output_images
               for tmp_dir in ["images", "labels"])

def test_aerial_dataset_loading(aerial_tile_size, aerial_config,
                                aerial_nb_labels, aerial_nb_output_images):
    """Load images into a AerialImage dataset
    """
    d = AerialDataset(aerial_tile_size)
    d.load(aerial_config)
    assert d.get_nb_labels() == aerial_nb_labels
    assert d.get_nb_images() == aerial_nb_output_images
