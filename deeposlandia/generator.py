"""Define data generator to feed Keras models
"""

import numpy as np

from keras.preprocessing.image import ImageDataGenerator


def feature_detection_labelling(img, label_ids):
    """One-hot encoding for feature detection problem

    Parameters
    ----------
    img : numpy.array
        Batched input image data of size (batch_size, image_size, image_size, 1)
    label_ids : list of integer
        Number of classes contained into the dataset
    Returns
    -------
    numpy.array
        Label encoding, array of shape (batch_size, nb_labels)
    """
    if not all(isinstance(item, (int, np.uint8, np.uint32, np.uint64)) for item in label_ids):
        raise ValueError(("List of label IDs must contains "
                          "integers: {}").format(label_ids))
    flattened_images = img.reshape(img.shape[0], -1).astype(np.uint8)
    one_hot_encoding = np.equal.outer(flattened_images, label_ids)
    return one_hot_encoding.any(axis=1)


def semantic_segmentation_labelling(img, label_ids):
    """One-hot encoder for semantic segmentation problem

    Parameters
    ----------
    img : ndarray
        Batched input image data of size (batch_size, image_size, image_size, 1)
    label_ids : list
        List of dataset label IDs

    Returns
    -------
    ndarray
        Label encoding, array of shape (batch_size, image_size, image_size, nb_labels), occurrence
    of the ith label for each pixel
    """
    if not all(isinstance(item, (int, np.uint8, np.uint32, np.uint64)) for item in label_ids):
        raise ValueError(("List of label IDs must contains "
                          "integers: {}").format(label_ids))
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
    seed : integer
        Random number generation for data shuffling and transformations

    Returns
    -------
    generator
        Input image generator
    """
    col_mode = 'grayscale' if gen_type == "labels" else 'rgb'
    generator = ImageDataGenerator()
    return generator.flow_from_directory(datapath,
                                         classes=[gen_type],
                                         target_size=(image_size, image_size),
                                         batch_size=batch_size,
                                         class_mode=None,
                                         color_mode=col_mode,
                                         seed=seed)

def create_generator(dataset, model, datapath, image_size, batch_size, label_ids,
                     inference=False, seed=None):
    """Create a Keras data Generator starting from images contained in `datapath` repository to
    address `model`

    Parameters
    ----------
    dataset : str
        Name of the dataset (*e.g.* `shapes` or `mapillary`)
    model : str
        Research problem that is addressed (either `feature_detection` or `semantic_segmentation`)
    datapath : str
        Path to image repository
    image_size : integer
        Number of width (resp. height) pixels
    batch_size : integer
        Number of images in each training batch
    label_ids : list of integer
        Dataset valid label IDs (background label are excluded)
    inference : boolean
        If True, generates only image data (labels are not considered during inference)
    seed : integer
        Random number generation for data shuffling and transformations

    Returns
    -------
    generator
        Generator of tuples (images, labels), for each input data batch

    """
    if not dataset in ['shapes', 'mapillary']:
        raise ValueError("Wrong dataset name {}".format(dataset))
    image_generator = feed_generator(datapath, "images", image_size, batch_size, seed)
    if inference:
        return image_generator
    label_generator = feed_generator(datapath, "labels", image_size, batch_size, seed)
    if model == 'feature_detection':
        label_generator = (feature_detection_labelling(x, label_ids)
                           for x in label_generator)
    elif model == 'semantic_segmentation':
        label_generator = (semantic_segmentation_labelling(x, label_ids)
                           for x in label_generator)
    else:
        raise ValueError("Wrong model name {}".format(model))
    return zip(image_generator, label_generator)
