"""Post-process predicted aerial tiles in order to rebuild complete high
resolution image.

Do inference on test images starting from a trained model and Keras API. Then
handle geographic datasets with dedicated libraries (GDAL, geopandas, shapely)
"""

import argparse
import glob
import os

import daiquiri
import geopandas as gpd
from osgeo import gdal
import numpy as np
from PIL import Image

from keras.models import Model
import keras.backend as K

from deeposlandia import utils, geometries
from deeposlandia.semantic_segmentation import SemanticSegmentationNetwork


logger = daiquiri.getLogger(__name__)


def get_image_paths(testing_folder, image_basename):
    """Returns a list with the path of every image that belongs to the
    `dataset`, preprocessed in `image_size`-pixelled images, that were
    extracted from an original image named as `image_basename`.

    Parameters
    ----------
    testing_folder : str
        Path of the testing image folder
    image_basename : str
        Original image name

    Returns
    -------
    list
        List of image full paths
    """
    image_raw_paths = os.path.join(testing_folder, "images", image_basename + "*")
    return [glob.glob(f) for f in [image_raw_paths]][0]


def extract_images(image_paths):
    """Convert a list of image filenames into a numpy array that contains the
    image data

    Parameters
    ----------
    image_paths : str
        Name of the image files onto the file system

    Returns
    -------
    np.array
        Data that is contained into the image
    """
    x_test = []
    for image_path in image_paths:
        if not image_path.endswith(".png"):
            logger.error("The filename does not refer to an png image.")
            raise ValueError()
        image = Image.open(image_path)
        if image.size[0] != image.size[1]:
            logger.error(
                "One of the parsed images has non-squared dimensions."
            )
            raise ValueError()
        x_test.append(np.array(image))
    return np.array(x_test)


def get_labels(datapath, dataset, tile_size):
    """Extract labels from the `dataset` glossary, according to the
    preprocessed version of the dataset

    Parameters
    ----------
    datapath : str
        Path of the data on the file system
    dataset : str
        Name of the dataset
    tile_size : int
        Size of preprocessed images, in pixels

    Returns
    -------
    list
        List of dictionnaries that describes the dataset labels
    """
    prepro_folder = utils.prepare_preprocessed_folder(
        datapath, dataset, tile_size,
    )
    if os.path.isfile(prepro_folder["testing_config"]):
        test_config = utils.read_config(prepro_folder["testing_config"])
    else:
        raise ValueError(
            (
                "There is no testing data with the given "
                "parameters. Please generate a valid dataset "
                "before calling the program."
            )
        )
    return [l for l in test_config["labels"] if l["is_evaluate"]]


def get_trained_model(model_filepath, image_size, nb_labels):
    """Recover model weights stored on the file system, and assign them into
    the `model` structure

    Parameters
    ----------
    model_filepath : str
        Path of the model on the file system
    image_size : int
        Image size, in pixels (height=width)
    nb_labels : int
        Number of output labels

    Returns
    -------
    keras.models.Model
        Convolutional neural network
    """
    K.clear_session()
    net = SemanticSegmentationNetwork(
        network_name="semseg_postprocessing",
        image_size=image_size,
        nb_labels=nb_labels,
        dropout=1.0,
        architecture="unet",
    )
    model = Model(net.X, net.Y)
    if os.path.isfile(model_filepath):
        model.load_weights(model_filepath)
        logger.info("Model weights have been recovered from %s" % model_filepath)
    else:
        logger.info(
            (
                "No available trained model for this image size"
                " with optimized hyperparameters. The "
                "inference will be done on an untrained model"
            )
        )
    return model


def assign_label_colors(predicted_labels, labels):
    """Transform raw Keras prediction into an exploitable numpy array that
    contains label IDs

    Parameters
    ----------
    predicted_labels : numpy.array
        Output of the prediction process, of shape (img_size, img_size)
    labels : list
        List of dictionnaries that describes the dataset labels

    Returns
    -------
    numpy.array
        Colored predicted labels, of shape (img_size, img_size, 3)
    """
    labelled_images = np.zeros(
        shape=np.append(predicted_labels.shape, 3), dtype=np.uint8
    )
    for idx, label in enumerate(labels):
        labelled_images[predicted_labels == idx] = label["color"]
    return labelled_images


def extract_coordinates_from_filenames(filenames):
    """Extract tiled image pixel coordinates starting from their filename, as
    it enrols the west and north coordinates for tile identification purpose

    Parameters
    ----------
    filenames : list
        List of tiled image filenames

    Returns
    -------
    list
        List of tiled image west and north pixel coordinates, regarding the
    full original image
    """
    basenames = [os.path.splitext(os.path.basename(f))[0] for f in filenames]
    coordinates = [b.split("_")[-2:] for b in basenames]
    return [[int(x), int(y)] for x, y in coordinates]


def fill_labelled_image(
    predictions, coordinates, tile_size, img_width, img_height=None
):
    """Fill labelled version of the original image with the labelled version of
    the corresponding tiles

    Parameters
    ----------
    predictions : numpy.array
        Labelled version of tiled images, according to semantic segmentation
    model predictions
    coordinates : list
        List of tiled image west and north pixel coordinates, regarding the
    full original image
    tile_size : int
        Tiled image size, in pixel
    img_width : int
        Original image width, in pixel
    img_height : int
        Original image height, in pixel

    Returns
    -------
    numpy.array
        Labelled version of the original image, where (i, j)-th pixel value
    corresponds to its predicted label
    """
    img_height = img_height if img_height is not None else img_width
    extended_width = img_width + tile_size - img_width % tile_size
    extended_height = img_height + tile_size - img_height % tile_size
    predicted_image = np.zeros(
        [extended_height, extended_width], dtype=np.uint8
    )
    for coords, image_data in zip(coordinates, predictions):
        x, y = coords
        predicted_image[y:(y + tile_size), x:(x + tile_size)] = image_data
    return predicted_image[:img_height, :img_width]


def build_full_labelled_image(
    images,
    coordinates,
    model,
    tile_size,
    img_width,
    img_height=None,
    batch_size=2,
):
    """Generate a full labelled version of an image (of shape (`img_width`,
        `img_height`)), knowing that it was previously splitted in tiles (of
        size `tile_size`). These tiles are stored on the file system as
        ̀image_paths`.

    This function includes a semantic segmentation model prediction step. The
    corresponding `model` is passed as another function argument, as well as
    the `batch_size` used during inference.

    Parameters
    ----------
    images : numpy.array
        Data associated to tiled images
    coordinates : list
        List of tiled image west and north pixel coordinates, regarding the
    full original image
    model : keras.models.Model
        Convolutional neural network
    tile_size : int
        Size of the tiled images, in pixel
    img_width : int
        Original image width, in pixel
    img_height : int
        Original image height, in pixel
    batch_size : int
        Number of images passed in each inference batches

    Returns
    -------
    numpy.array
        Labelled version of the original image, where (i, j)-th pixel value
    corresponds to its predicted label
    """
    img_height = img_height if img_height is not None else img_width
    y_raw_preds = model.predict(images, batch_size=batch_size, verbose=1)
    predicted_labels = np.argmax(y_raw_preds, axis=3)
    full_labelled_image = fill_labelled_image(
        predicted_labels, coordinates, tile_size, img_width, img_height
    )
    return full_labelled_image


def draw_grid(data, img_width, img_height, tile_size):
    """Draw a white grid on an original image labelled version, depending on
    its tile splitting, *i.e.* draw vertical and horizontal lines each
    `tile_size` pixels.

    Parameters
    ----------
    data : numpy.array
        Labelled version of an image
    img_width : int
        Original image width, in pixel
    img_height : int
        Original image height, in pixel
    tile_size : int
        Size of the tiled images, in pixel

    Returns
    -------
    numpy.array
        Labelled version of an image, with an explicit white grid that
    highlights the tile splitting
    """
    gridded_data = data.copy()
    for i in range(tile_size, img_width, tile_size):
        gridded_data[:, i] = np.full([img_height, 3], 255, dtype=np.uint8)
    for i in range(tile_size, img_height, tile_size):
        gridded_data[i] = np.full([img_width, 3], 255, dtype=np.uint8)
    return gridded_data


def get_image_features(datapath, dataset, filename):
    """Retrieve geotiff image features with GDAL

    Use the `GetGeoTransform` method, that provides the following values:
      + East/West location of Upper Left corner
      + East/West pixel resolution
      + 0.0
      + North/South location of Upper Left corner
      + 0.0
      + North/South pixel resolution

    A GDAL dataset is opened during the function execution. The corresponding
    variable is set to None at the end of the function so as to free memory.

    See GDAL documentation (https://www.gdal.org/gdal_tutorial.html)

    Parameters
    ----------
    datapath : str
    dataset : str
    filename : str
        Name of the image file from which coordinates are extracted

    Returns
    -------
    dict
        Bounding box of the image (west, south, east, north coordinates), srid,
    and size (in pixels)

    """
    input_folder = utils.prepare_input_folder(datapath, dataset)
    filepath = os.path.join(
        input_folder, "testing", "images", filename + ".tif"
    )
    ds = gdal.Open(filepath)
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + height * gt[5]
    maxx = gt[0] + width * gt[1]
    maxy = gt[3]
    srid = int(ds.GetProjection().split('"')[-2])
    ds = None  # Free memory used by the GDAL Dataset
    return {
        "west": minx,
        "south": miny,
        "east": maxx,
        "north": maxy,
        "srid": srid,
        "width": width,
        "height": height,
    }


def main(args):

    logger.info("Postprocess %s...", args.image_basename)
    features = get_image_features(
        args.datapath, args.dataset, args.image_basename
    )

    img_width, img_height = features["width"], features["height"]
    logger.info("Raw image size: %s, %s" % (img_width, img_height))

    prepro_folder = utils.prepare_preprocessed_folder(args.datapath, args.dataset, args.image_size)
    image_paths = get_image_paths(prepro_folder["testing"], args.image_basename)
    logger.info("The image will be splitted into %s tiles" % len(image_paths))
    images = extract_images(image_paths)
    coordinates = extract_coordinates_from_filenames(image_paths)
    labels = get_labels(args.datapath, args.dataset, args.image_size)

    output_folder = utils.prepare_output_folder(
        args.datapath, args.dataset, args.image_size, "semseg"
    )
    model = get_trained_model(
        output_folder["best-model"], args.image_size, len(labels)
    )

    logger.info("Predict labels for input images...")
    data = build_full_labelled_image(
        images,
        coordinates,
        model,
        args.image_size,
        img_width,
        img_height,
        args.batch_size,
    )
    logger.info(
        "Labelled image dimension: %s, %s" % (data.shape[0], data.shape[1])
    )
    colored_data = assign_label_colors(data, labels)
    if args.draw_grid:
        colored_data = draw_grid(
            colored_data, img_width, img_height, args.image_size
        )
    predicted_label_file = os.path.join(
        output_folder["labels"],
        args.image_basename + "_" + str(args.image_size) + ".png",
    )
    Image.fromarray(colored_data).save(predicted_label_file)

    vectorized_labels, vectorized_data = geometries.vectorize_mask(
        data, colored_data, labels
    )
    gdf = gpd.GeoDataFrame(
        {"labels": vectorized_labels, "geometry": vectorized_data}
    )
    predicted_geom_file = os.path.join(
        output_folder["geometries"],
        args.image_basename + "_" + str(args.image_size) + ".geojson",
    )
    if os.path.isfile(predicted_geom_file):
        os.remove(predicted_geom_file)
    gdf.to_file(predicted_geom_file, driver="GeoJSON")

    rasterized_data = geometries.rasterize_polygons(
        vectorized_data, vectorized_labels, img_height, img_width
    )
    colored_raster_data = assign_label_colors(rasterized_data, labels)
    if args.draw_grid:
        colored_raster_data = draw_grid(
            colored_raster_data, img_width, img_height, args.image_size
        )
    predicted_raster_file = os.path.join(
        output_folder["rasters"],
        args.image_basename + "_" + str(args.image_size) + ".png",
    )
    Image.fromarray(colored_raster_data).save(predicted_raster_file)
