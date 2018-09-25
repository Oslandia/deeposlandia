"""Define data generator to feed Keras models
"""

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

def recover_label_id(img, label_config):
    """Recover label ids starting from a typical RGB-colored image modeled as a
    (width, height, 3)-shaped array

    Parameters
    ----------
    img : np.array
        Image data of shape (width, heigth, 3)
    label_config : dict
        Labels contained into the dataset

    Returns
    -------
    np.array
        Array of pixel labels of shape (width, heigth)
    """
    reshaped_img = img.reshape([-1, 3])
    label_img = np.full(shape=reshaped_img.shape[0],
                        fill_value=-1,
                        dtype=reshaped_img.dtype)
    for label in label_config:
        label_img[np.all(label['color']==reshaped_img, axis=1)] = label['id']
    return label_img.reshape(img.shape[:3])


def feature_detection_labelling(img, label_config):
    """One-hot encoding for feature detection problem

    Parameters
    ----------
    img : numpy.array
        Batched input image data of size (batch_size, image_size, image_size, 1)
    label_config : dict
        Labels contained into the dataset

    Returns
    -------
    numpy.array
        Label encoding, array of shape (batch_size, nb_labels)
    """
    if not (img.shape[1] == img.shape[2] and len(img.shape) == 4):
        raise ValueError(("Wrong image shape. Please provide batched RGB-"
                          "images with equal width and height dimensions."))
    label_ids = [item['id'] for item in label_config if item['is_evaluate']]
    if not all(isinstance(item, (int, np.uint8, np.uint32, np.uint64)) for item in label_ids):
        raise ValueError(("List of label IDs must contains "
                          "integers: {}").format(label_ids))
    img = recover_label_id(img, label_config);
    flattened_images = img.reshape(img.shape[0], -1).astype(np.uint8)
    one_hot_encoding = np.equal.outer(flattened_images, label_ids)
    return one_hot_encoding.any(axis=1)


def semantic_segmentation_labelling(img, label_config):
    """One-hot encoder for semantic segmentation problem

    Parameters
    ----------
    img : ndarray
        Batched input image data of size (batch_size, image_size, image_size, 1)
    label_config : dict
        Label contained into the dataset

    Returns
    -------
    ndarray
        Label encoding, array of shape (batch_size, image_size, image_size, nb_labels), occurrence
    of the ith label for each pixel
    """
    if not (img.shape[1] == img.shape[2] and len(img.shape) == 4):
        raise ValueError(("Wrong image shape. Please provide batched RGB-"
                          "images with equal width and height dimensions."))
    label_ids = [item['id'] for item in label_config if item['is_evaluate']]
    if not all(isinstance(item, (int, np.uint8, np.uint32, np.uint64)) for item in label_ids):
        raise ValueError(("List of label IDs must contains "
                          "integers: {}").format(label_ids))
    img = recover_label_id(img, label_config);
    img = img.squeeze().astype(np.uint8)
    one_hot_encoding = np.equal.outer(img, label_ids)
    return one_hot_encoding


def feed_generator(datapath, gen_type, image_size, batch_size, seed=None):
    """Build a couple of generator fed by image and label repository, respectively

    The input image are stored as RGB-images, whilst labelled image are grayscaled-images. The
    Keras generator takes this difference into account through its `color_mode` parameter, that
    depends on `gen_type`

    Parameters
    ----------
    datapath : str
        Path to image repository
    gen_type : str
        Generator type, either `images` or `labels`
    image_size : integer
        Number of width (resp. height) pixels
    batch_size : integer
        Number of images in each training batch
    they are stored as RGB pixels
    seed : integer
        Random number generation for data shuffling and transformations

    Returns
    -------
    generator
        Input image generator
    """
    generator = ImageDataGenerator()
    return generator.flow_from_directory(datapath,
                                         classes=[gen_type],
                                         target_size=(image_size, image_size),
                                         batch_size=batch_size,
                                         class_mode=None,
                                         color_mode='rgb',
                                         seed=seed)

def create_generator(dataset, model, datapath, image_size, batch_size, label_config,
                     inference=False, seed=None):
    """Create a Keras data Generator starting from images contained in `datapath` repository to
    address `model`

    Parameters
    ----------
    dataset : str
        Name of the dataset (*e.g.* `shapes`, `mapillary` or `aerial`)
    model : str
        Research problem that is addressed (either `feature_detection` or `semantic_segmentation`)
    datapath : str
        Path to image repository
    image_size : integer
        Number of width (resp. height) pixels
    batch_size : integer
        Number of images in each training batch
    label_config : dict
        Dataset valid label description
    inference : boolean
        If True, generates only image data (labels are not considered during inference)
    seed : integer
        Random number generation for data shuffling and transformations

    Returns
    -------
    generator
        Generator of tuples (images, labels), for each input data batch

    """
    if not dataset in ['shapes', 'mapillary', 'aerial']:
        raise ValueError("Wrong dataset name {}".format(dataset))
    image_generator = feed_generator(datapath, "images", image_size,
                                     batch_size, seed)
    if inference:
        return image_generator
    label_generator = feed_generator(datapath, "labels", image_size,
                                     batch_size, seed)
    if model == 'feature_detection':
        label_generator = (feature_detection_labelling(x, label_config)
                           for x in label_generator)
    elif model == 'semantic_segmentation':
        label_generator = (semantic_segmentation_labelling(x, label_config)
                           for x in label_generator)
    else:
        raise ValueError("Wrong model name {}".format(model))
    return zip(image_generator, label_generator)
