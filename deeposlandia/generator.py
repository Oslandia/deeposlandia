"""Define data generator to feed Keras models
"""

import numpy as np

import keras as K
from keras.preprocessing.image import ImageDataGenerator

def to_categorical(img, label_ids):
    """One-hot encoder for a labelled image

    Parameters
    ----------
    img : ndarray
        2D or 3D (if batched)
    label_ids : list
        List of dataset label IDs

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

def create_generator(dataset, datapath, image_size, batch_size, config, seed=1337):
    """Create a Keras data Generator starting from images contained in `datapath` repository

    Parameters
    ---------- 
    dataset : str
        Name of the dataset
    datapath : str
        Path to image repository
    image_size : integer
        Number of width (resp. height) pixels
    batch_size : integer
        Number of images in each training batch
    config : dict
        Dataset glossary
    seed : integer
        Random number generation for data shuffling and transformations

    Returns
    -------
    generator
        Generator of tuples (images, labels), for each input data batch
    """
    generator = ImageDataGenerator()
    im_generator = generator.flow_from_directory(
        datapath,
        classes=['images'],
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode=None,
        seed=seed)
    label_generator = generator.flow_from_directory(
        datapath,
        classes=['labels'],
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode=None,
        color_mode='grayscale',
        seed=seed)
    if dataset == 'shapes':
        label_ids = [c for c in config['classes']]
        label_ids.sort()
    else:
        labels = config['labels']
        label_ids = [x['id'] for x in labels]
    label_generator = iter(to_categorical(x, label_ids)
                           for x in label_generator)
    return zip(im_generator, label_generator)
