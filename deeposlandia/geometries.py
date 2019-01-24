"""Set of functions dedicated to georeferenced object handling
"""

import json
import os

import cv2
import daiquiri
import fiona
import geojson
import geopandas as gpd
import numpy as np
from osgeo import osr
import shapely.geometry as shgeom


logger = daiquiri.getLogger(__name__)


def get_pixel(coord, min_coord, max_coord, size):
    """Transform abscissa from geographical coordinate to pixel

    For horizontal operations, 'min_coord', 'max_coord' and 'size' refer
    respectively to west and east coordinates and image width.

    For vertical operations, 'min_coord', 'max_coord' and 'size' refer
    respectively to north and south coordinates and image height.

    Parameters
    ----------
    coord : list
        Coordinates to transform
    min_coord : float
        Georeferenced minimal coordinate of the image
    max_coord : float
        Georeferenced maximal coordinate of the image
    size : int
        Image size, in pixels

    Returns
    -------
    list
        Transformed coordinates, as pixel references within the image
    """
    if isinstance(coord, list):
        return [
            int(size * (c - min_coord) / (max_coord - min_coord))
            for c in coord
        ]
    elif isinstance(coord, float):
        return int(size * (coord - min_coord) / (max_coord - min_coord))
    else:
        raise TypeError(
            "Unknown type (%s), pass a 'list' or a 'float'", type(coord)
        )


def get_geocoord(coord, min_coord, max_coord, size):
    """Transform abscissa from pixel to geographical coordinate

    For horizontal operations, 'min_coord', 'max_coord' and 'size' refer
    respectively to west and east coordinates and image width.

    For vertical operations, 'min_coord', 'max_coord' and 'size' refer
    respectively to north and south coordinates and image height.

    Parameters
    ----------
    coord : list
        Coordinates to transform
    min_coord : float
        Minimal coordinates of the image, as a pixel reference
    max_coord : float
        Maximal coordinates of the image, as a pixel reference
    size : int
        Image size, in pixels

    Returns
    -------
    list
        Transformed coordinates, expressed in the accurate coordinate system
    """
    if isinstance(coord, list):
        return [min_coord + c * (max_coord - min_coord) / size for c in coord]
    elif isinstance(coord, int):
        return min_coord + coord * (max_coord - min_coord) / size
    else:
        raise TypeError(
            "Unknown type (%s), pass a 'list' or a 'int'", type(coord)
        )


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
    return {
        "west": minx,
        "south": miny,
        "east": maxx,
        "north": maxy,
        "srid": srid,
        "width": width,
        "height": height,
    }


def extract_points_from_polygon(p, geofeatures, min_x, min_y):
    """Extract points from a polygon

    Parameters
    ----------
    p : shapely.geometry.Polygon
        Polygon to detail
    geofeatures : dict
        Geographical features associated to the image
    min_x : int
        Minimal x-coordinate (west)
    min_y : int
        Minimal y-coordinate (north)
    Returns
    -------
    numpy.array
        Polygon vertices

    """
    raw_xs, raw_ys = p.exterior.xy
    xs = get_pixel(
        list(raw_xs),
        geofeatures["west"],
        geofeatures["east"],
        geofeatures["width"]
    )
    ys = get_pixel(
        list(raw_ys),
        geofeatures["north"],
        geofeatures["south"],
        geofeatures["height"]
    )
    points = np.array([[y, x] for x, y in zip(xs, ys)], dtype=np.int32)
    points[:, 0] -= min_y
    points[:, 1] -= min_x
    return points


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
    max_x = min_x + tile_width
    max_y = min_y + tile_height
    min_x_coord, max_x_coord = get_geocoord(
        [min_x, max_x], features["west"], features["east"], features["width"]
    )
    min_y_coord, max_y_coord = get_geocoord(
        [min_y, max_y],
        features["north"],
        features["south"],
        features["height"]
    )
    return shgeom.Polygon(
        (
            (min_x_coord, min_y_coord),
            (max_x_coord, min_y_coord),
            (max_x_coord, max_y_coord),
            (min_x_coord, max_y_coord),
        )
    )


def extract_tile_items(
    raster_features, labels, min_x, min_y, tile_width, tile_height
):
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

    Returns
    -------
    geopandas.GeoDataFrame
        Set of ground-truth labels contained into the tile, characterized by
    their type (complete, unfinished or foundation) and their geometry

    """
    area = get_tile_footprint(
        raster_features, min_x, min_y, tile_width, tile_height
    )
    bdf = gpd.GeoDataFrame(
        crs=fiona.crs.from_epsg(raster_features["srid"]), geometry=[area]
    )
    reproj_labels = labels.to_crs(epsg=raster_features["srid"])
    tile_items = gpd.sjoin(reproj_labels, bdf)
    if tile_items.shape[0] == 0:
        return tile_items[["condition", "geometry"]]
    tile_items = gpd.overlay(tile_items, bdf)
    tile_items = tile_items.explode()  # Manage MultiPolygons
    return tile_items[["condition", "geometry"]]
