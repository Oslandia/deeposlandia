"""Unit test related to data folder organization
"""

import os

from deeposlandia.utils import (
    prepare_input_folder,
    prepare_preprocessed_folder,
    prepare_output_folder,
)


def test_input_folder(datapath_repo):
    """Test the existence of the raw dataset directory when using
    ̀utils.prepare_input_folder`

    """
    datapath = str(datapath_repo)
    dataset = "shapes"
    prepare_input_folder(datapath, dataset)
    assert os.path.isdir(os.path.join(datapath, dataset, "input"))


def test_preprocessed_folder(datapath_repo):
    """Test the creation of the preprocessed data repositories, by checking the
    full expected tree, *i.e.* considering training, validation and testing
    repositories within an instance-specific folder, and images and labels
    repositories wihtin each of these subrepositories

    """
    datapath = str(datapath_repo)
    dataset = "shapes"
    image_size = 64
    aggregate = "full"
    prepare_preprocessed_folder(datapath, dataset, image_size, aggregate)
    assert os.path.isdir(os.path.join(datapath, dataset, "preprocessed"))
    assert os.path.isdir(
        os.path.join(
            datapath,
            dataset,
            "preprocessed",
            str(image_size) + "_" + aggregate,
        )
    )
    assert os.path.isdir(
        os.path.join(
            datapath,
            dataset,
            "preprocessed",
            str(image_size) + "_" + aggregate,
            "training",
        )
    )
    assert os.path.isdir(
        os.path.join(
            datapath,
            dataset,
            "preprocessed",
            str(image_size) + "_" + aggregate,
            "training",
            "images",
        )
    )
    assert os.path.isdir(
        os.path.join(
            datapath,
            dataset,
            "preprocessed",
            str(image_size) + "_" + aggregate,
            "training",
            "labels",
        )
    )
    assert os.path.isdir(
        os.path.join(
            datapath,
            dataset,
            "preprocessed",
            str(image_size) + "_" + aggregate,
            "validation",
        )
    )
    assert os.path.isdir(
        os.path.join(
            datapath,
            dataset,
            "preprocessed",
            str(image_size) + "_" + aggregate,
            "validation",
            "images",
        )
    )
    assert os.path.isdir(
        os.path.join(
            datapath,
            dataset,
            "preprocessed",
            str(image_size) + "_" + aggregate,
            "validation",
            "labels",
        )
    )
    assert os.path.isdir(
        os.path.join(
            datapath,
            dataset,
            "preprocessed",
            str(image_size) + "_" + aggregate,
            "testing",
        )
    )
    assert os.path.isdir(
        os.path.join(
            datapath,
            dataset,
            "preprocessed",
            str(image_size) + "_" + aggregate,
            "testing",
            "images",
        )
    )


def test_output_folder(datapath_repo):
    """Test the creation of the dataset output repository, by checking the full
    expected tree, *i.e.* until the instance-specific checkpoint folder

    """
    datapath = str(datapath_repo)
    dataset = "shapes"
    model = "feature_detection"
    instance_name = "test_instance"
    prepare_output_folder(datapath, dataset, model, instance_name)
    assert os.path.isdir(os.path.join(datapath, dataset, "output"))
    assert os.path.isdir(os.path.join(datapath, dataset, "output", model))
    assert os.path.isdir(
        os.path.join(datapath, dataset, "output", model, "checkpoints")
    )
    assert os.path.isdir(
        os.path.join(
            datapath, dataset, "output", model, "checkpoints", instance_name
        )
    )
