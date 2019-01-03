"""Tanzania building dataset class

To add building instances into the dataset, use the load_buildings() method by
passing as the first parameter the folder that contains building informations
(images, features, items)

"""

import json
from multiprocessing import Pool
import os

import cv2
import daiquiri
import fiona
import geopandas as gpd
import numpy as np
from osgeo import gdal
from PIL import Image
import shapely.geometry as shgeom

from deeposlandia.datasets import Dataset
from deeposlandia import utils


logger = daiquiri.getLogger(__name__)

# Save png tiles without auxiliary information on disk
os.environ['GDAL_PAM_ENABLED'] = 'NO'

class TanzaniaDataset(Dataset):
    """Tanzania building dataset, as released during the Open AI Tanzania
    challenge

    See:
    https://blog.werobotics.org/2018/08/06/welcome-to-the-open-ai-tanzania-challenge/
    The dataset is composed of 20 high-resolution images (~6-8cm/pixel, until
    76k*76k pixels), 13 being associated with geo-referenced labels, 7 for
    testing purpose.

    Attributes
    ----------
    img_size : int
        Size of the tiles into which each raw images is decomposed during
    dataset population (height=width)

    """

    BACKGROUND_COLOR = [0, 0, 0]
    COMPLETE_COLOR = [50, 200, 50]
    INCOMPLETE_COLOR = [200, 200, 50]
    FOUNDATION_COLOR = [200, 50, 50]

    def __init__(self, img_size):
        """Class constructor ; instanciates a TanzaniaDataset as a standard Dataset
        which is completed by a glossary file that describes the dataset labels
        and images

        """
        super().__init__(img_size)
        self.add_label(label_id=0, label_name="background",
                       color=self.BACKGROUND_COLOR, is_evaluate=True)
        self.add_label(label_id=1, label_name="complete",
                       color=self.COMPLETE_COLOR, is_evaluate=True)
        self.add_label(label_id=2, label_name="incomplete",
                       color=self.INCOMPLETE_COLOR, is_evaluate=True)
        self.add_label(label_id=3, label_name="foundation",
                       color=self.FOUNDATION_COLOR, is_evaluate=True)


    def _generate_preprocessed_filenames(
            self, image_filename, output_dir, x, y, suffix
    ):
        """Generate preprocessed image and label filenames on the file system,
        starting from a raw image filename

        Parameters
        ----------
        image_filename : str
            Original image filename
        output_dir : str
            Output folder for preprocessed material
        x : int
            Extracted image west coordinates
        y : int
            Extracted image north coordinates
        suffix : str
            Preprocessed filename complement

        Returns
        -------
        dict
            Preprocessed image and corresponding label filenames
        """
        basename_decomp = os.path.splitext(os.path.basename(image_filename))
        img_id_str = (str(self.image_size) + '_'
                      + str(self.image_size) + '_'
                      + str(x) + '_' + str(y) + "_" + suffix)
        new_filename = basename_decomp[0] + '_' + img_id_str + ".png"
        out_image_name = os.path.join(output_dir, 'images', new_filename)
        out_label_name = out_image_name.replace("images", "labels")
        return {"image": out_image_name, "labels": out_label_name}


    def _serialize(
            self, tile_image, labelled_image, label_dict,
            image_filename, output_dir, x, y, suffix
    ):
        """Serialize a tiled image generated from an original high-resolution
        raster as well as the labelled version of the tile

        The method returns a dict that contains image-related file paths.

        Parameters
        ----------
        tile_image : PIL.Image
        labelled_image : PIL.Image
        label_dict : dict
        image_filename : str
        output_dir : str
        x : int
        y : int

        Returns
        -------
        dict
            Information related to the serialized tile (file paths, encountered labels)
        """
        dirs = self._generate_preprocessed_filenames(
            image_filename, output_dir, x, y, suffix
        )
        # gdal.Translate(dirs["image"], raster,
        #                format="PNG",
        #                srcWin=[x, y, self.image_size, self.image_size])
        tile_image.save(dirs["image"])
        labelled_image.save(dirs["labels"])
        return {"raw_filename": image_filename,
                "image_filename": dirs["image"],
                "label_filename": dirs["labels"],
                "labels": label_dict}


    def _preprocess_tile(self, x, y, image_filename, output_dir,
                         raster, labels=None):
        """Preprocess one single tile built from `image_filename`, with respect
                         to pixel coordinates `(x, y)`

        Parameters
        ----------
        x : int
            Horizontal pixel coordinate (*i.e.* west bound)
        y : int
            Vertical pixel coordinate (*i.e.* north bound)
        image_filename : str
            Full path towards the image on the disk
        output_dir : str
            Output path where preprocessed image must be saved
        raster : osgeo.gdal.Dataset
            Original georeferenced raster
        labels : geopandas.GeoDataFrame
            Raw image labels (*i.e.* georeferenced buildings)

        Returns
        -------
        dict
            Key/values with the filenames and label ids

        """
        dirs = self._generate_preprocessed_filenames(
            image_filename, output_dir, x, y
        )
        gdal.Translate(dirs["image"], raster,
                       format="PNG",
                       srcWin=[x, y, self.image_size, self.image_size])
        return {"raw_filename": image_filename,
                "image_filename": dirs["image"]}


    def _preprocess_for_inference(self, image_filename, output_dir):
        """Resize/crop then save the training & label images

        Parameters
        ----------
        image_filename : str
            Full path towards the image on the disk
        output_dir : str
            Output path where preprocessed image must be saved

        Returns
        -------
        dict
            Key/values with the filenames and label ids
        """
        raster = gdal.Open(image_filename)
        raw_img_width = raster.RasterXSize
        raw_img_height = raster.RasterYSize
        result_dicts = []
        logger.info("Raw image size: %s, %s" % (raw_img_width, raw_img_height))
        logger.info("Image filename: %s" % image_filename)

        for x in range(0, raw_img_width, self.image_size):
            for y in range(0, raw_img_height, self.image_size):
                tile_results = self._preprocess_tile(x, y, image_filename,
                                                     output_dir, raster)
                result_dicts.append(tile_results)
        del raster
        return result_dicts


    def _preprocess_for_training(self, image_filename, output_dir, nb_images):
        """Resize/crop then save the training & label images

        Parameters
        ----------
        image_filename : str
            Full path towards the image on the disk
        output_dir : str
            Output path where preprocessed image must be saved

        Returns
        -------
        dict
            Key/values with the filenames and label ids
        """
        raster = gdal.Open(image_filename)
        raw_img_width = raster.RasterXSize
        raw_img_height = raster.RasterYSize
        image_data = raster.ReadAsArray()
        image_data = np.swapaxes(image_data, 0, 2)
        print(image_data.shape, raw_img_width, raw_img_height)
        # image_data = np.swapaxes(image_data, 0, 2)
        result_dicts = []
        logger.info("Raw image size: %s, %s" % (raw_img_width, raw_img_height))
        logger.info("Image filename: %s" % image_filename)

        label_filename = (image_filename
                          .replace("images", "labels")
                          .replace(".tif", ".geojson"))
        labels = gpd.read_file(label_filename)
        labels = labels.loc[~labels.geometry.isna(), ["condition", "geometry"]]
        none_mask = [lc is None for lc in labels.condition]
        labels.loc[none_mask, "condition"] = "Complete"

        nb_attempts = 0
        image_counter = 0
        empty_image_counter = 0
        while image_counter < nb_images and nb_attempts < 3 * nb_images:
            # randomly pick an image
            x = np.random.randint(0, raw_img_width - self.image_size)
            y = np.random.randint(0, raw_img_height - self.image_size)

            tile_data = image_data[x:(x+self.image_size),
                                   y:(y+self.image_size)]
            tile_image = Image.fromarray(tile_data)
            raster_features = get_image_features(raster)
            tile_items = extract_tile_items(raster_features, labels,
                                            x, y,
                                            self.image_size,
                                            self.image_size,
                                            tile_srid=32737)
            mask = self.load_mask(tile_items, raster_features, x, y)
            label_dict = utils.build_labels(mask,
                                            range(self.get_nb_labels()),
                                            "tanzania")
            labelled_image = utils.build_image_from_config(mask, self.labels)
            if len(tile_items) > 0:
                tiled_results = self._serialize(
                    tile_image, labelled_image, label_dict,
                    image_filename, output_dir, x, y, "nw"
                )
                result_dicts.append(tiled_results)
                image_counter += 1
                tile_image_ne = tile_image.transpose(Image.FLIP_LEFT_RIGHT)
                labelled_image_ne = labelled_image.transpose(Image.FLIP_LEFT_RIGHT)
                tiled_results_ne = self._serialize(
                    tile_image_ne, labelled_image_ne, label_dict,
                    image_filename, output_dir, x, y, "ne"
                )
                result_dicts.append(tiled_results_ne)
                image_counter += 1
                tile_image_sw = tile_image.transpose(Image.FLIP_TOP_BOTTOM)
                labelled_image_sw = labelled_image.transpose(Image.FLIP_TOP_BOTTOM)
                tiled_results_sw = self._serialize(
                    tile_image_sw, labelled_image_sw, label_dict,
                    image_filename, output_dir, x, y, "sw"
                )
                result_dicts.append(tiled_results_sw)
                image_counter += 1
                tile_image_se = tile_image_sw.transpose(Image.FLIP_LEFT_RIGHT)
                labelled_image_se = labelled_image_sw.transpose(Image.FLIP_LEFT_RIGHT)
                tiled_results_se = self._serialize(
                    tile_image_se, labelled_image_se, label_dict,
                    image_filename, output_dir, x, y, "se"
                )
                result_dicts.append(tiled_results_se)
                image_counter += 1
                del tile_image_se, tile_image_sw, tile_image_ne
                del labelled_image_se, labelled_image_sw, labelled_image_ne
            else:
                if empty_image_counter < 0.1 * nb_images:
                    tiled_results = self._serialize(
                        tile_image, labelled_image, label_dict,
                        image_filename, output_dir, x, y, "nw"
                        )
                    result_dicts.append(tiled_results)
                    image_counter += 1
                    empty_image_counter += 1
            nb_attempts += 1
        del raster
        logger.info("Generate %s images after %s attempts."
                    % (image_counter, nb_attempts))
        return result_dicts


    def populate(self, output_dir, input_dir, nb_images=None,
                 aggregate=False, labelling=True):
        """ Populate the dataset with images contained into `datadir` directory

        Parameters
        ----------
        output_dir : str
            Path of the directory where the preprocessed image must be saved
        input_dir : str
            Path of the directory that contains input images
        nb_images : integer
            Number of images to be considered in the dataset; if None, consider the whole
        repository
        aggregate : bool
            Label aggregation parameter, useless for this dataset, but kept for
        class method genericity
        labelling: boolean
            If True labels are recovered from dataset, otherwise dummy label are generated
        """
        image_list = os.listdir(os.path.join(input_dir, "images"))
        image_list_longname = [os.path.join(input_dir, "images", l)
                               for l in image_list
                               if not l.startswith('.')]
        nb_image_files = len(image_list_longname)

        logger.info("Getting %s images to preprocess..."
                    % nb_image_files)
        logger.info(image_list_longname)
        if labelling:
            nb_tile_per_image = int(nb_images/nb_image_files)
            for x in image_list_longname:
                self.image_info.append(self._preprocess_for_training(x,
                                                                     output_dir, nb_tile_per_image))
            # with Pool() as p:
            #     self.image_info = p.starmap(self._preprocess_for_training,
            #                                 [(x, output_dir, nb_tile_per_image)
            #                                  for x in image_list_longname])
        else:
            with Pool() as p:
                self.image_info = p.starmap(self._preprocess_for_inference,
                                            [(x, output_dir)
                                             for x in image_list_longname])

        self.image_info = [item for sublist in self.image_info
                           for item in sublist]
        logger.info("Saved %s images in the preprocessed dataset."
                    % len(self.image_info))


    def load_mask(self, buildings, raster_features, min_x, min_y):
        """Translate georeferenced buildings as numpy arrays in order to
        prepare image analysis

        Parameters
        ----------
        buildings : geopandas.GeoDataFrame
            Georeferenced building labels, with a ̀condition` column that
        contains building type and a `geometry` column that describe the
        geolocalization
        raster_features : dict
            Geographical features of raw original image
        min_x : int
            Minimal tile x-coordinates (west bound)
        min_y : int
            Minimal tile y-coordinates (north bound)

        Returns
        -------
        numpy.array
            Array-versionned building labels, that link each pixel to the
        specified class, *i.e.* `B(x, y)=i` if pixel `(x, y)` belongs to class
        `i`. In this dataset, the labels are `complete`, `incomplete` or
        ̀foundation`.

        """
        mask = np.zeros(shape=(self.image_size,
                               self.image_size),
                        dtype=np.uint8)
        if buildings.shape[0] == 0:
            return mask
        for idx, row in buildings.iterrows():
            points = self.extract_points_from_polygon(row["geometry"],
                                                      raster_features)
            points[:, 0] -= min_x
            points[:, 1] -= min_y
            label_id = [label["id"] for label in self.labels
                        if label["name"] == row["condition"].lower()][0]
            mask = cv2.fillPoly(mask, [points], label_id)
        return mask


    def extract_points_from_polygon(self, p, features):
        """Extract points from a polygon

        Parameters
        ----------
        p : shapely.geometry.Polygon
            Polygon to detail
        features : dict
            Geographical features associated to the image
        Returns
        -------
        np.array
            Polygon vertices

        """
        raw_xs, raw_ys = p.exterior.xy
        xs = get_x_pixel(raw_xs, features["east"], features["west"], features["width"])
        ys = get_y_pixel(raw_ys, features["south"], features["north"], features["height"])
        points = np.array([[x, y] for x, y in zip(xs, ys)], dtype=np.int32)
        return points


def get_x_pixel(coord, east, west, width):
    """Transform abscissa from geographical coordinate to pixel

    Parameters
    ----------
    coord : list
        Coordinates to transform
    east : float
        East coordinates of the image
    west : float
        West coordinates of the image
    width : int
        Image width
    Returns
    -------
    list
        Transformed X-coordinates
    """
    return [int(width * (west-c) / (west-east)) for c in coord]


def get_y_pixel(coord, south, north, height):
    """Transform abscissa from geographical coordinate to pixel

    Parameters
    ----------
    coord : list
        Coordinates to transform
    south : float
        South coordinates of the image
    north : float
        North coordinates of the image
    height : int
        Image height

    Returns
    -------
    list
        Transformed Y-coordinates
    """
    return [int(height * (north-c) / (north-south)) for c in coord]


def get_x_geocoord(coord, east, west, width):
    """Transform abscissa from pixel to geographical coordinate

    Parameters
    ----------
    coord : list
        Coordinates to transform
    east : float
        East coordinates of the image
    west : float
        West coordinates of the image
    width : int
        Image width
    Returns
    -------
    list
        Transformed X-coordinates
    """
    return west + coord * (east-west) / width


def get_y_geocoord(coord, south, north, height):
    """Transform abscissa from pixel to geographical coordinate

    Parameters
    ----------
    coord : list
        Coordinates to transform
    south : float
        South coordinates of the image
    north : float
        North coordinates of the image
    height : int
        Image height

    Returns
    -------
    list
        Transformed Y-coordinates
    """
    return north + coord * (south-north) / height


def get_image_features(raster):
    """Retrieve geotiff image features with GDAL

    Use the `GetGeoTransform` method, that provides the following values:
        + East/West location of Upper Left corner
        + East/West pixel resolution
        + 0.0
        + North/South location of Upper Left corner
        + 0.0
        + North/South pixel resolution

    See GDAL documentation (https://www.gdal.org/gdal_tutorial.html)

    Parameters
    ----------
    raster : osgeo.gdal.Dataset
        Active opened image as a GDAL object

    Returns
    -------
    dict
        Bounding box of the image (west, south, east, north coordinates), srid,
        and size (in pixels)

    """
    width = raster.RasterXSize
    height = raster.RasterYSize
    gt = raster.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + height * gt[5]
    maxx = gt[0] + width * gt[1]
    maxy = gt[3]
    srid = int(raster.GetProjection().split('"')[-2])
    return {"west": minx, "south": miny, "east": maxx, "north": maxy,
            "srid": srid, "width": width, "height": height}


def get_tile_footprint(features, min_x, min_y, tile_width, tile_height=None):
    """Compute a tile geographical footprint expressed as a `shapely` geometry
    that contains geographical coordinates of tile corners

    Parameters
    ----------
    features : dict
        Raw image raster geographical features (`north`, `south`, `east` and
    `west` coordinates, `weight` and `height` measured in pixels)
    min_x : int
        Left tile limit, as a horizontal pixel index
    min_y : int
        Upper tile limit, as a vertical pixel index
    tile_width : int
        Tile width, measured in pixel
    tile_height : int
        Tile height, measured in pixel: if None, consider
    `tile_height=tile_width` (squared tile)

    Returns
    -------
    shapely.geometry.Polygon
        Tile footprint, as a square polygon delimited by its corner
    geographical coordinates

    """
    tile_height = tile_width if tile_height is None else tile_height
    min_x_coord = get_x_geocoord(min_x, features["east"],
                                 features["west"], features["width"])
    min_y_coord = get_y_geocoord(min_y, features["south"],
                                 features["north"], features["height"])
    max_x_coord = get_x_geocoord(min_x + tile_width, features["east"],
                                 features["west"], features["width"])
    max_y_coord = get_y_geocoord(min_y + tile_height, features["south"],
                                 features["north"], features["height"])
    return shgeom.Polygon(((min_x_coord, min_y_coord),
                           (max_x_coord, min_y_coord),
                           (max_x_coord, max_y_coord),
                           (min_x_coord, max_y_coord)))


def extract_tile_items(raster_features, labels, min_x, min_y,
                       tile_width, tile_height, tile_srid):
    """Extract label items that belong to the tile defined by the minimum
    horizontal pixel `min_x` (left tile limit), the minimum vertical pixel
    `min_y` (upper tile limit) and the sizes ̀tile_width` and `tile_height`
    measured as a pixel amount.

    The tile is cropped from the original image raster as follows:
      - horizontally, between `min_x` and `min_x+tile_width`
      - vertically, between `min_y` and `min_y+tile_height`

    This method takes care of original data projection (UTM 37S, Tanzania
    area), however this parameter may be changed if similar data on another
    projection is used.

    Parameters
    ----------
    raster_features : dict
        Raw image raster geographical features (`north`, `south`, `east` and
    `west` coordinates, `weight` and `height` measured in pixels)
    labels : geopandas.GeoDataFrame
        Raw image labels, as a set of geometries
    min_x : int
        Left tile limit, as a horizontal pixel index
    min_y : int
        Upper tile limit, as a vertical pixel index
    tile_width : int
        Tile width, measured in pixel
    tile_height : int
        Tile height, measured in pixel
    tile_srid : int
        Ground-truth label projection, as an EPSG code (ex: 32737, for UTM37S
    area)

    Returns
    -------
    geopandas.GeoDataFrame
        Set of ground-truth labels contained into the tile, characterized by
    their type (complete, unfinished or foundation) and their geometry

    """
    area = get_tile_footprint(raster_features, min_x, min_y,
                              tile_width, tile_height)
    bdf = gpd.GeoDataFrame(crs=fiona.crs.from_epsg(tile_srid),
                           geometry=[area])
    reproj_labels = labels.to_crs(epsg=tile_srid)
    tile_items = gpd.sjoin(reproj_labels, bdf)
    if tile_items.shape[0] == 0:
        return tile_items[["condition", "geometry"]]
    tile_items = gpd.overlay(tile_items, bdf)
    tile_items = tile_items.explode() # Manage MultiPolygons
    return tile_items[["condition", "geometry"]]
