"""Unit test related to the dataset creation, population and loading
"""

import json
import os
import shutil

from deeposlandia.dataset import Dataset, MapillaryDataset, ShapeDataset

def test_dataset_creation():
    """Create a generic dataset
    """
    IMAGE_SIZE = 128
    d = Dataset(IMAGE_SIZE)
    assert d.image_size == IMAGE_SIZE
    assert d.get_nb_labels() == 0
    assert d.get_nb_images() == 0
    d.add_label(label_id=0, label_name="test_label", color=[100, 100, 100], is_evaluate=True)
    assert d.get_nb_labels() == 1

def test_mapillary_dataset_creation():
    """Create a Mapillary dataset from a configuration file
    """
    IMAGE_SIZE = 224
    config_filename = "tests/data/mapillary/config_aggregate.json"
    with open(config_filename) as fobj:
        config = json.load(fobj)
    NB_LABELS = len(config["labels"])
    d = MapillaryDataset(IMAGE_SIZE, config_filename)
    assert d.image_size == IMAGE_SIZE
    assert d.get_nb_labels(see_all=True) == NB_LABELS
    assert d.get_nb_images() == 0

def test_mapillary_dataset_population(tmpdir):
    """Populate a Mapillary dataset
    """
    IMAGE_SIZE = 224
    config_filename = "tests/data/mapillary/config_aggregate.json"
    with open(config_filename) as fobj:
        config = json.load(fobj)
    NB_LABELS = len(config["labels"])
    input_dir = "tests/data/mapillary/input_dataset_population/"
    NB_IMAGES = len(os.listdir(os.path.join(input_dir, "images")))
    output_dir = str(tmpdir.realpath())
    tmpdir.mkdir("images")
    tmpdir.mkdir("labels")
    d = MapillaryDataset(IMAGE_SIZE, config_filename)
    d.populate(output_dir, input_dir, nb_images=NB_IMAGES)
    d.save(output_dir + '.json')
    assert d.get_nb_labels(see_all=True) == NB_LABELS
    assert d.get_nb_images() == NB_IMAGES
    assert os.path.isfile(output_dir + '.json')
    assert all(len(os.listdir(str(tmp_dir))) == NB_IMAGES
               for tmp_dir in tmpdir.listdir())

def test_mapillary_dataset_loading():
    """Load images into a Mapillary dataset
    """
    IMAGE_SIZE = 224
    config_filename = "tests/data/mapillary/config_aggregate.json"
    dataset_filename = "tests/data/mapillary/training.json"
    image_files = os.listdir("tests/data/mapillary/training/images")
    NB_IMAGES = len(image_files)
    with open(config_filename) as fobj:
        config = json.load(fobj)
    NB_LABELS = len(config["labels"])
    d = MapillaryDataset(IMAGE_SIZE, config_filename)
    d.load(dataset_filename)
    assert d.get_nb_labels(see_all=True) == NB_LABELS
    assert d.get_nb_images() == NB_IMAGES

def test_shape_dataset_creation():
    """Create a Shapes dataset
    """
    IMAGE_SIZE = 64
    NB_LABELS = 4
    d = ShapeDataset(IMAGE_SIZE)
    assert d.image_size == IMAGE_SIZE
    assert d.get_nb_labels(see_all=True) == NB_LABELS
    assert d.get_nb_images() == 0

def test_shape_dataset_population(tmpdir):
    """Populate a Shapes dataset
    """
    IMAGE_SIZE = 64
    NB_LABELS = 4
    NB_IMAGES = 20
    output_dir = str(tmpdir.realpath())
    tmpdir.mkdir("images")
    tmpdir.mkdir("labels")
    d = ShapeDataset(IMAGE_SIZE)
    d.populate(output_dir, nb_images=NB_IMAGES)
    d.save(output_dir + '.json')
    assert d.get_nb_labels(see_all=True) == NB_LABELS
    assert d.get_nb_images() == NB_IMAGES
    assert os.path.isfile(output_dir + '.json')
    assert all(len(os.listdir(str(tmp_dir))) == NB_IMAGES
               for tmp_dir in tmpdir.listdir())

def test_shape_dataset_loading():
    """Load images into a Shapes dataset
    """
    IMAGE_SIZE = 64
    NB_LABELS = 4
    dataset_filename = "tests/data/shapes/training.json"
    image_files = os.listdir("tests/data/shapes/training/images")
    NB_IMAGES = len(image_files)
    d = ShapeDataset(IMAGE_SIZE)
    d.load(dataset_filename)
    assert d.get_nb_labels(see_all=True) == NB_LABELS
    assert d.get_nb_images() == NB_IMAGES
