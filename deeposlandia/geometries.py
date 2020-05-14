"""Set of functions dedicated to georeferenced object handling

The functions which allow to convert raster in vector and vice-versa are
inspired from:
  - https://www.kaggle.com/lopuhin/full-pipeline-demo-poly-pixels-ml-poly

"""

from collections import defaultdict

import cv2
import daiquiri
from fiona.crs import from_epsg
import geopandas as gpd
import numpy as np
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
        geofeatures["width"],
    )
    ys = get_pixel(
        list(raw_ys),
        geofeatures["north"],
        geofeatures["south"],
        geofeatures["height"],
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
        features["height"],
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
        crs=from_epsg(raster_features["srid"]), geometry=[area]
    )
    reproj_labels = labels.to_crs(epsg=raster_features["srid"])
    tile_items = gpd.sjoin(reproj_labels, bdf)
    if tile_items.shape[0] == 0:
        return tile_items[["condition", "geometry"]]
    tile_items = gpd.overlay(tile_items, bdf)
    tile_items = tile_items.explode()  # Manage MultiPolygons
    return tile_items[["condition", "geometry"]]


def extract_geometry_vertices(mask, structure_size=(10, 10), approx_eps=0.01):
    """Extract polygon vertices from a boolean mask with the help of OpenCV
    utilities, as a numpy array

    Parameters
    ----------
    mask : numpy.array
        Image mask where to find polygons
    structure_size : tuple
        Size of the cv2 structuring artefact, as a tuple of horizontal and
    vertical pixels
    approx_eps : double
        Approximation coefficient, aiming at building more simplified polygons
    (this coefficient lies between 0 and 1, the larger the value is, the more
    important the approximation is)

    Returns
    -------
    numpy.array
        List of polygons contained in the mask, identified by their vertices
    """
    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, structure_size)
    denoised = cv2.morphologyEx(mask, cv2.MORPH_OPEN, structure)
    grown = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, structure)
    _, contours, hierarchy = cv2.findContours(
        grown, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = [
        cv2.approxPolyDP(
            c, epsilon=approx_eps * cv2.arcLength(c, closed=True), closed=True
        )
        for c in contours
    ]
    return polygons, hierarchy


def retrieve_area_color(data, contour, labels):
    """Mask an image area and retrieve its dominant color starting from a label
    glossary, by determining its closest label (regarding euclidean distance).

    Largely inspired from : https://www.pyimagesearch.com/\
    2016/02/15/determining-object-color-with-opencv/

    Parameters
    ----------
    data : np.array
        3-channelled image
    contour : np.array
        List of points that delimits the area
    labels : list
        List of dictionnary that describes each labels (with "id" and "color" keys)
    """
    mask = np.zeros(data.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean = cv2.mean(data, mask=mask)[:3]
    min_dist = (np.inf, None)
    for label in labels:
        d = np.linalg.norm(label["color"] - np.array(mean))
        if d < min_dist[0]:
            min_dist = (d, label["id"])
    return min_dist[1]


def vectorize_mask(
    mask, colored_data, labels,
    min_area=10.0, structure_size=(10, 10), approx_eps=0.01
):
    """Convert a numpy array (*i.e.* rasterized information) into a shapely
    Multipolygon (*i.e.* vectorized information), by filtering too small
    objects. As a reminder, this function is supposed to be used for building
    detection, hence by definition a building can not be smaller than a few
    square meters

    A large part of the function (and comments) comes from:
      - https://www.kaggle.com/lopuhin/full-pipeline-demo-poly-pixels-ml-poly

    See also
    http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html

    Parameters
    ----------
    mask : numpy.array
        1-channelled version of the input image
    data : numpy.array
        3-channelled version of the input  image
    labels : list of dicts
        List of labels, as dicts that contain at least a "id"
    min_area : double
    structure_size : tuple
        Size of the cv2 structuring artefact, as a tuple of horizontal and
    vertical pixels
    approx_eps : double
        Approximation coefficient, aiming at building more simplified polygons
    (this coefficient lies between 0 and 1, the larger the value is, the more
    important the approximation is)

    Returns
    -------
    shapely.geometry.MultiPolygon
        Set of detected objects grouped as a MultiPolygon object
    """
    contours, hierarchy = extract_geometry_vertices(
        mask, structure_size, approx_eps
    )
    if not contours:
        return np.array([]), shgeom.MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    out_labels = []
    for idx, cnt in enumerate(contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = shgeom.Polygon(
                shell=cnt[:, 0, :],
                holes=[
                    c[:, 0, :]
                    for c in cnt_children.get(idx, [])
                    if cv2.contourArea(c) >= min_area
                ],
            )
            poly_label = retrieve_area_color(colored_data, cnt, labels)
            all_polygons.append(poly)
            out_labels.append(poly_label)
    all_polygons = shgeom.MultiPolygon(all_polygons)
    # approximating polygons might have created invalid ones, fix them
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == "Polygon":
            all_polygons = shgeom.MultiPolygon([all_polygons])
    return np.array(out_labels), all_polygons


def rasterize_polygons(polygons, labels, img_height, img_width):
    """Transform a vectorized information into a numpy mask for plotting
    purpose

    Inspired from:
      - https://www.kaggle.com/lopuhin/full-pipeline-demo-poly-pixels-ml-poly

    Parameters
    ----------
    polygons : shapely.geometry.MultiPolygon
        Set of detected objects, stored as a MultiPolygon
    labels : np.array
        List of corresponding labels that describes polygon classes
    img_height : int
        Image height, in pixels
    img_width : int
        Image width, in pixels

    Returns
    -------
    numpy.array
        Rasterized polygons
    """
    img_mask = np.zeros(shape=(img_height, img_width), dtype=np.uint8)
    if not polygons:
        return img_mask
    for polygon, label in zip(polygons, labels):
        exterior = np.array(polygon.exterior.coords).round().astype(np.int32)
        interiors = [
            np.array(pi.coords).round().astype(np.int32)
            for pi in polygon.interiors
        ]
        cv2.fillPoly(img_mask, [exterior], int(label))
        cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


def pixel_to_geocoord(polygon_ring, geofeatures):
    """Transform point coordinates from pixel to geographical coordinates in
    the accurate geographical projection, *i.e.* the projection of original
    image indicated in the `geofeatures` dict

    Parameters
    ----------
    polygon_ring : shapely.geometry.polygon.LinearRing
        Sequence of points belonging to the object of interest
    geofeatures : dict
        Geographical characteristics of the original image

    Returns
    -------
    numpy.array
        Transformed sequence of points, expressed in geographical coordinates,
    regarding the SRID provided by `geofeatures["srid"]`

    """
    xpixels, ypixels = polygon_ring.xy
    px = get_geocoord(
        list(xpixels),
        geofeatures["west"],
        geofeatures["east"],
        geofeatures["width"],
    )
    py = get_geocoord(
        list(ypixels),
        geofeatures["north"],
        geofeatures["south"],
        geofeatures["height"],
    )
    return np.array([px, py]).T


def convert_to_geocoord(polygons, geofeatures):
    """Convert pixel-defined polygons into georeferenced polygon with the
    projection of the original raster, provided by `geofeatures["srid"]`

    Parameters
    ----------
    polygons : shapely.geometry.MultiPolygon or list of
    shapely.geometry.Polygons
        Detected polygons identified with their vertex pixels on the image
    geofeatures : dict
        Geographical features associated to the original image

    Returns
    -------
    shapely.geometry.MultiPolygon
        Georeferenced multipolygon that describes every object belonging to
    original image
    """
    geopolygons = []
    for polygon in polygons:
        exterior = pixel_to_geocoord(polygon.exterior, geofeatures)
        interiors = [
            pixel_to_geocoord(pi, geofeatures) for pi in polygon.interiors
        ]
        if len(interiors) > 0:
            poly = shgeom.Polygon(shell=exterior, holes=[i for i in interiors])
        else:
            poly = shgeom.Polygon(shell=exterior)
        geopolygons.append(poly)
    return geopolygons
