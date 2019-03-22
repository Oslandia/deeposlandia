"""Unit test related to model training with Keras API
"""

import numpy as np
import os

from keras.models import Model

from deeposlandia.generator import create_generator
from deeposlandia.feature_detection import FeatureDetectionNetwork
from deeposlandia.utils import read_config


def test_model_training(
    shapes_image_size,
    shapes_sample,
    shapes_sample_config,
    shapes_temp_dir,
    shapes_nb_images,
):
    """Test the training of a simple neural network with Keras API, as well as
    model inference and trained model backup

    One big test function to avoid duplicating the training operations (that
    can be long)
    """
    BATCH_SIZE = 10
    NB_EPOCHS = 1
    NB_STEPS = shapes_nb_images // BATCH_SIZE
    config = read_config(shapes_sample_config)
    label_ids = [x["id"] for x in config["labels"] if x["is_evaluate"]]
    gen = create_generator(
        "shapes",
        "featdet",
        shapes_sample,
        shapes_image_size,
        BATCH_SIZE,
        config["labels"],
    )
    cnn = FeatureDetectionNetwork(
        "test", image_size=shapes_image_size, nb_labels=len(label_ids)
    )
    model = Model(cnn.X, cnn.Y)
    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["acc"]
    )
    hist = model.fit_generator(gen, epochs=NB_EPOCHS, steps_per_epoch=NB_STEPS)
    assert len(hist.history) == 2
    assert all(k in hist.history.keys() for k in ["acc", "loss"])
    assert hist.history["acc"][0] >= 0 and hist.history["acc"][0] <= 1

    test_image = np.random.randint(
        0, 255, [BATCH_SIZE, shapes_image_size, shapes_image_size, 3]
    )
    score = model.predict(test_image)
    assert score.shape == (BATCH_SIZE, len(label_ids))
    assert all(0 <= s and s <= 1 for s in score.ravel())

    BACKUP_FILENAME = os.path.join(
        str(shapes_temp_dir),
        "checkpoints",
        "test_model_{:02d}.h5".format(NB_EPOCHS),
    )
    model.save(BACKUP_FILENAME)
    assert os.path.isfile(BACKUP_FILENAME)


def test_model_backup_loading(
    shapes_image_size, shapes_sample_config, shapes_temp_dir
):
    """Test the model checkpoint recovering
    """
    config = read_config(shapes_sample_config)
    label_ids = [x["id"] for x in config["labels"] if x["is_evaluate"]]

    cnn = FeatureDetectionNetwork(
        "test", image_size=shapes_image_size, nb_labels=len(label_ids)
    )
    model = Model(cnn.X, cnn.Y)
    old_weights = model.get_weights()
    checkpoint_path = os.path.join(str(shapes_temp_dir), "checkpoints")
    if os.path.isdir(checkpoint_path):
        checkpoints = os.listdir(checkpoint_path)
        if len(checkpoints) > 0:
            model_checkpoint = max(checkpoints)
            trained_model_epoch = int(model_checkpoint[-5:-3])
            checkpoint_complete_path = os.path.join(
                checkpoint_path, model_checkpoint
            )
            model.load_weights(checkpoint_complete_path)
        else:
            trained_model_epoch = 0
    else:
        trained_model_epoch = 0
    new_weights = model.get_weights()
    assert trained_model_epoch > 0
    assert len(old_weights) == len(new_weights)
    assert old_weights[0].shape == new_weights[0].shape
    # Test if old and new weights are different (at least for one layer)
    assert any(
        not np.allclose(lhs, rhs) for lhs, rhs in zip(old_weights, new_weights)
    )
