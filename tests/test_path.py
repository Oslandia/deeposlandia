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
    prepare_preprocessed_folder(datapath, dataset, image_size)
    assert os.path.isdir(os.path.join(datapath, dataset, "preprocessed"))
    assert os.path.isdir(
        os.path.join(
            datapath,
            dataset,
            "preprocessed",
            str(image_size),
        )
    )
    assert os.path.isdir(
        os.path.join(
            datapath,
            dataset,
            "preprocessed",
            str(image_size),
            "training",
        )
    )
    assert os.path.isdir(
        os.path.join(
            datapath,
            dataset,
            "preprocessed",
            str(image_size),
            "training",
            "images",
        )
    )
    assert os.path.isdir(
        os.path.join(
            datapath,
            dataset,
            "preprocessed",
            str(image_size),
            "training",
            "labels",
        )
    )
    assert os.path.isdir(
        os.path.join(
            datapath,
            dataset,
            "preprocessed",
            str(image_size),
            "validation",
        )
    )
    assert os.path.isdir(
        os.path.join(
            datapath,
            dataset,
            "preprocessed",
            str(image_size),
            "validation",
            "images",
        )
    )
    assert os.path.isdir(
        os.path.join(
            datapath,
            dataset,
            "preprocessed",
            str(image_size),
            "validation",
            "labels",
        )
    )
    assert os.path.isdir(
        os.path.join(
            datapath,
            dataset,
            "preprocessed",
            str(image_size),
            "testing",
        )
    )
    assert os.path.isdir(
        os.path.join(
            datapath,
            dataset,
            "preprocessed",
            str(image_size),
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
    image_size = 100
    model = "feature_detection"
    output_folder = prepare_output_folder(datapath, dataset, image_size, model)
    assert len(output_folder.keys()) == 6
    assert os.path.isdir(
        os.path.join(datapath, dataset, "output", model, "checkpoints")
    )
    assert os.path.isdir(
        os.path.join(datapath, dataset, "output", model, "predicted_labels")
    )
    dataset = "aerial"
    prepare_output_folder(datapath, dataset, image_size, model)
    assert os.path.isdir(
        os.path.join(datapath, dataset, "output", model, "predicted_geometries")
    )
    assert os.path.isdir(
        os.path.join(datapath, dataset, "output", model, "predicted_rasters")
    )
