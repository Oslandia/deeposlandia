"""Unit test related to model training with Keras API
"""

import numpy as np
import os

from keras.utils.test_utils import keras_test
from keras.models import Model

from deeposlandia.generator import create_generator
from deeposlandia.keras_feature_detection import FeatureDetectionNetwork
from deeposlandia.utils import read_config

@keras_test
def test_model_training(shapes_image_size, shapes_training_data, shapes_config, shapes_nb_images):
    """Test the training of a simple neural network with Keras API, as well as model inference and
    trained model backup

    One big test function to avoid duplicating the training operations (that can be long)
    """
    BATCH_SIZE = 10
    NB_STEPS = shapes_nb_images // BATCH_SIZE
    config = read_config(str(shapes_config))
    label_ids = [x['id'] for x in config["labels"]]

    gen = create_generator("shapes", "feature_detection", str(shapes_training_data),
                           shapes_image_size, BATCH_SIZE, label_ids)
    cnn = FeatureDetectionNetwork("test", image_size=shapes_image_size, nb_labels=len(label_ids))
    model = Model(cnn.X, cnn.Y)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    hist = model.fit_generator(gen, epochs=1, steps_per_epoch=NB_STEPS)
    assert len(hist.history) == 2
    assert all(k in hist.history.keys() for k in ['acc', 'loss'])
    assert hist.history['acc'][0] >= 0 and hist.history['acc'][0] <= 1

    test_image = np.random.randint(0, 255, [BATCH_SIZE, shapes_image_size, shapes_image_size, 3])
    score = model.predict(test_image)
    assert score.shape == (BATCH_SIZE, len(label_ids))
    assert all(0 <= s and s <= 1 for s in score.ravel())

    BACKUP_FILENAME = os.path.join(str(shapes_training_data), "checkpoints", "test_model.h5")
    model.save(BACKUP_FILENAME)
    assert os.path.isfile(BACKUP_FILENAME)
