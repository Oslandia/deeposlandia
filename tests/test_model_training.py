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
def test_model_training():
    """Test the training of a simple neural network with Keras API, as well as model inference and
    trained model backup

    One big test function to avoid duplicating the training operations (that can be long)
    """
    IMAGE_SIZE = 64
    BATCH_SIZE = 10
    dataset = "shapes"
    model = "feature_detection"
    datapath = ("./tests/data/" + dataset + "/training")
    config = read_config(datapath + ".json")
    NB_IMAGES = len(os.listdir(os.path.join(datapath, "images")))
    NB_STEPS = NB_IMAGES // BATCH_SIZE
    print(NB_STEPS)
    label_ids = [x['id'] for x in config["labels"]]

    gen = create_generator(dataset, model, datapath, IMAGE_SIZE, BATCH_SIZE, label_ids)
    cnn = FeatureDetectionNetwork("test", image_size=IMAGE_SIZE, nb_labels=len(label_ids))
    model = Model(cnn.X, cnn.Y)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    hist = model.fit_generator(gen, epochs=1, steps_per_epoch=NB_STEPS)
    assert len(hist.history) == 2
    assert all(k in hist.history.keys() for k in ['acc', 'loss'])
    assert hist.history['acc'][0] >= 0 and hist.history['acc'][0] <= 1

    test_image = np.random.randint(0, 255, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    score = model.predict(test_image)
    assert score.shape == (BATCH_SIZE, len(label_ids))
    assert all(0 <= s and s <= 1 for s in score.ravel())

    BACKUP_PATH = "./tests/data/" + dataset + "/backups/"
    os.makedirs(BACKUP_PATH, exist_ok=True)
    BACKUP_FILENAME = os.path.join(BACKUP_PATH, "test_model.h5")
    model.save(BACKUP_FILENAME)
    assert os.path.isfile(BACKUP_FILENAME)
