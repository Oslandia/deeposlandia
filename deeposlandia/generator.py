"""Define data generator to feed Keras models
"""

import numpy as np

import keras as K
from keras.preprocessing.image import ImageDataGenerator

def find_classes(img, config):
    """Convert RGB image data into a matrix which contains the glossary class that correspond to
    each pixel (-1 if no object)

    As images have been stored with OpenCV, the color format is (B,G,R). The values in the last
    dimension must then be inverted (`img[i, x, y, ::-1]` operation).

    Parameters
    ----------
    img : np.array
        Input RGB images
    config : dict
        Dataset glossary

    Returns
    -------
    np.array
        Output array of shape (batch_size, img_size, img_size, nb_classes) with the index of each
    pixel class stored at each pixel position
    """
    colors = [config['classes'][sc]['color'] for sc in config['classes']]
    nb_classes = len(config['classes'])
    a = np.zeros(img.shape[:3]) + nb_classes
    for class_id in range(nb_classes):
        for i in range(img.shape[0]):
            a[i][[[img[i, x, y, ::-1].tolist() == colors[class_id] for y in range(img.shape[2])]
                  for x in range(img.shape[1])]] = class_id
    return a

def to_categorical(img, config, dataset):
    """One-hot encoder for a labelled image

    Parameters
    ----------
    img : ndarray
        2D or 3D (if batched)
    config : dict
        Dataset glossary
    dataset : str
        Name of the dataset

    Returns
    -------
    ndarray
        Occurrence of the ith label for each pixel
    """
    if dataset == 'shapes':
        img = find_classes(img, config)
        label_ids = list(config["classes"].keys())
    else:
        labels = config["labels"]
        label_ids = [x['id'] for x in labels]
    img = img.squeeze()
    input_shape = img.shape
    img = img.ravel().astype(np.uint8)
    for idx, label in enumerate(label_ids):
        print(idx, label)
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
    if dataset == 'shapes':
        label_generator = generator.flow_from_directory(
            datapath,
            classes=['labels'],
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode=None,
            seed=seed)
    else:
        label_generator = generator.flow_from_directory(
            datapath,
            classes=['labels'],
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode=None,
            color_mode='grayscale',
            seed=seed)
    label_generator = iter(to_categorical(x, config, dataset)
                           for x in label_generator)
    return zip(im_generator, label_generator)
