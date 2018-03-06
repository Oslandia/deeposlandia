"""A first draft neural network with Keras for semantic segmentation (Mapillary)

This script uses the aggregated labels of Mapillary, i.e. 66 classes turn into 11 classes.
"""

import os
import json
import os.path as osp
from datetime import datetime

import numpy as np

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import (Input, Conv2D, MaxPool2D, Conv2DTranspose,
                          Activation, BatchNormalization)
from keras.models import Model


_now = datetime.now()
SIZE = 256
BATCH_SIZE = 32
# number of images per epoch
NB_IMAGES = 18000
NB_VALIDATION_IMAGES = 2000
STEPS = NB_IMAGES // BATCH_SIZE
print("STEP {}".format(STEPS))
VALIDATION_STEPS = NB_VALIDATION_IMAGES // BATCH_SIZE
print("VALIDATION STEPS {}".format(VALIDATION_STEPS))
CLASSES = 11
EPOCHS = 30
DATADIR = "../data/mapillary"
OUTPUT_DIR = osp.join(DATADIR, 'segnet-output', 'run-{}'.format(_now.strftime("%Y-%m-%dT%H:%M")))
OUTPUT_DIR = OUTPUT_DIR
CONFIG = osp.join(DATADIR, 'config_aggregate.json')
TRAINING = osp.join(DATADIR, "training_aggregate_{}".format(SIZE))
VALIDATION = osp.join(DATADIR, "validation_aggregate_{}".format(SIZE))
SEED = 1337


os.makedirs(OUTPUT_DIR, exist_ok=True)


def to_categorical(img, label_ids):
    """One-hot encoder for a labelled image

    Parameters
    ----------
    img : ndarray
        2D or 3D (if batched)

    Returns
    -------
    ndarray
        Occurrence of the ith label for each pixel
    """
    img = img.squeeze()
    input_shape = img.shape
    img = img.ravel().astype(np.uint8)
    for idx, label in enumerate(label_ids):
        mask = img == label
        img[mask] = idx
    n = img.shape[0]
    categorical = np.zeros((n, len(label_ids)))
    categorical[np.arange(n), img] = 1
    output_shape = input_shape + (len(label_ids), )
    return categorical.reshape(output_shape)


def block_encoder(x, num_filters, kernel_size=3):
    """Encoder block

    Convolution + Batch normalization + ReLU activation + Max Pooling

    Parameters
    ----------
    x : keras.Layer
    num_filters : int
    kernel_size : int (3 by default)

    Returns
    -------
    keras.Layer
    """
    x = Conv2D(num_filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(2)(x)
    return x


def block_decoder(x, num_filters, kernel_size=3):
    """Decoder block.

    For now, just a Convolution Transpose

    Parameters
    ----------
    x : keras.Layer
    num_filters : int
    kernel_size : int (3 by default)

    Returns
    -------
    keras.Layer
    """
    x = Conv2DTranspose(num_filters, padding='same', activation='relu',
                        kernel_size=kernel_size, strides=2)(x)
    return x


def read_config():
    """Juste read the JSON configuration file.
    """
    with open(CONFIG) as fobj:
        return json.load(fobj)


config = read_config()
labels = config['labels']
label_ids = np.array([x['id'] for x in labels])

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

images_generator = train_datagen.flow_from_directory(
    TRAINING,
    classes=['images'],
    target_size=(SIZE, SIZE),
    batch_size=BATCH_SIZE,
    class_mode=None,
    seed=SEED)
masks_generator = test_datagen.flow_from_directory(
    TRAINING,
    classes=['labels'],
    target_size=(SIZE, SIZE),
    batch_size=BATCH_SIZE,
    class_mode=None,
    color_mode='grayscale',
    seed=SEED)
validation_images_generator = train_datagen.flow_from_directory(
    VALIDATION,
    classes=['images'],
    target_size=(SIZE, SIZE),
    batch_size=BATCH_SIZE,
    class_mode=None,
    seed=SEED)
validation_mask_generator = train_datagen.flow_from_directory(
    VALIDATION,
    classes=['labels'],
    target_size=(SIZE, SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode=None,
    seed=SEED)
masks_generator = iter(to_categorical(x, label_ids) for x in masks_generator)
train_generator = zip(images_generator, masks_generator)
validation_mask_generator = iter(to_categorical(x, label_ids)
                                 for x in validation_mask_generator)
validation_generator = zip(validation_images_generator, validation_mask_generator)

# Input layer. Squarred RGB image.
img_input = Input(shape=(SIZE, SIZE, 3))

# encoder part
x = block_encoder(img_input, 32)
x = block_encoder(x, 64)
x = block_encoder(x, 128)
x = block_encoder(x, 256)

# decoder part
x = block_decoder(x, 256)
x = block_decoder(x, 128)
x = block_decoder(x, 64)
x = block_decoder(x, 32)
# Number of feature maps equals to the number of classes, i.e. labels
x = Conv2DTranspose(CLASSES, kernel_size=2, padding='same')(x)

# Last layer
y = Activation('softmax', name='target')(x)

model = Model(img_input, y)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc', 'mae'])
json_model = model.to_json()
with open(osp.join(OUTPUT_DIR, "segnet-model-keras.json"), "w") as fobj:
    fobj.write(json_model)
print(model.summary())
print("See output dir {}".format(OUTPUT_DIR))
tsboard = keras.callbacks.TensorBoard(log_dir=osp.join(OUTPUT_DIR, 'log'), write_graph=True)
check = keras.callbacks.ModelCheckpoint(
    osp.join(OUTPUT_DIR, "checkpoint-segnet-epoch-{epoch:02d}-valacc-{val_acc:.2f}.h5"),
    monitor='val_acc',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto', period=1)
earlystop = keras.callbacks.EarlyStopping(monitor='val_acc',
                                          min_delta=0.002,
                                          patience=10,
                                          verbose=0,
                                          mode='auto')
hist = model.fit_generator(train_generator, epochs=EPOCHS,
                           callbacks=[tsboard, check, earlystop],
                           steps_per_epoch=STEPS,
                           validation_data=validation_generator,
                           validation_steps=VALIDATION_STEPS)
metrics = {"epoch": hist.epoch,
           "metrics": hist.history,
           "params": hist.params}
with open(osp.join(OUTPUT_DIR, "hist-metrics.json"), "w") as fobj:
    json.dump(metrics, fobj)
fname = "final-segnet-batch-{}-size-{}-max-epoch-{}.h5"
fname = fname.format(BATCH_SIZE, SIZE, EPOCHS)
model.save(osp.join(OUTPUT_DIR, fname))
