"""Unit tests that address geometries.py module functions
"""

import os
import pytest

import geopandas as gpd
import numpy as np
from osgeo import gdal
from shapely.geometry import Polygon

from deeposlandia.datasets.tanzania import (
    extract_points_from_polygon, extract_tile_items,
    get_geocoord, get_image_features, get_pixel, get_tile_footprint
)


def test_get_pixel():
    """Test the transformation from georeferenced coordinates to pixel
    """
    geocoords = [10000.0, 15000.0, 20000.0]
    min_coord, max_coord = 0, 30000.0
    size_in_pixel = 500
    true_pixel_coord = [166, 250, 333]
    pixel_coord = get_pixel(
        geocoords, min_coord, max_coord, size_in_pixel
    )
    assert np.all(pixel_coord == true_pixel_coord)
    single_geocoord = geocoords[0]
    single_pixel_coord = get_pixel(
        single_geocoord, min_coord, max_coord, size_in_pixel
    )
    assert single_pixel_coord == true_pixel_coord[0]
    str_geocoord = "15000"
    with pytest.raises(TypeError):
        get_pixel(str_geocoord, min_coord, max_coord, size_in_pixel)


def test_get_geocoord():
    """Test the transformation from pixel to georeferenced coordinates
    """
    pixels = [166, 250, 333]
    min_coord, max_coord = 0, 30000.0
    size_in_pixel = 500
    true_geocoord = [9960.0, 15000.0, 19980.0]
    geocoord = get_geocoord(pixels, min_coord, max_coord, size_in_pixel)
    assert np.all(geocoord == true_geocoord)
    single_pixel = pixels[0]
    single_geocoord = get_geocoord(
        single_pixel, min_coord, max_coord, size_in_pixel
    )
    assert single_geocoord == true_geocoord[0]
    str_pixel = "250"
    with pytest.raises(TypeError):
        get_geocoord(str_pixel, min_coord, max_coord, size_in_pixel)


def test_get_image_features(tanzania_example_image):
    """Test the image geographic feature recovering
    """
    ds = gdal.Open(str(tanzania_example_image))
    geofeatures = get_image_features(ds)
    assert isinstance(geofeatures["south"], float)
    assert isinstance(geofeatures["north"], float)
    assert isinstance(geofeatures["east"], float)
    assert isinstance(geofeatures["west"], float)
    assert geofeatures["west"] < geofeatures["east"]
    assert geofeatures["south"] < geofeatures["north"]
    assert isinstance(geofeatures["srid"], int)
    assert isinstance(geofeatures["width"], int)
    assert isinstance(geofeatures["height"], int)
    assert geofeatures["width"] > 0
    assert geofeatures["height"] > 0


def test_extract_points_from_polygon(tanzania_example_image):
    """Test a polygon point extraction

    Within a 1000x1000 pixel original image, consider 500x500 tiles, and more
    specifically the right-bottom tile. One wants to retrieve a triangle
    whose coordinates are as follows:
    - (image_width/2, image_height/2)
    - (image_width/2, image_height)
    - (image_width*3/4, image_height/2)

    The point coordinate representation must be inverted between georeferenced
    points and 2D-'numpy.array' pixel points: in the latter, the first
    (resp. the second) dimension corresponds to rows (resp.columns).
    """
    ds = gdal.Open(str(tanzania_example_image))
    geofeatures = get_image_features(ds)
    min_x = min_y = 500
    x1 = geofeatures["west"] + (geofeatures["east"] - geofeatures["west"]) / 2
    y1 = (
        geofeatures["south"]
        + (geofeatures["north"] - geofeatures["south"]) / 2
    )
    x2 = x1 + (geofeatures["east"] - x1) / 2
    y2 = geofeatures["south"]
    polygon = Polygon(
        ((x1, y1), (x1, y2), (x2, y1), (x1, y1))
    )
    points = extract_points_from_polygon(polygon, geofeatures, min_x, min_y)
    expected_points = np.array([[0, 0], [500, 0], [0, 250], [0, 0]])
    print(points)
    print(polygon.exterior)
    print(expected_points)
    assert np.all(points == expected_points)


def test_square_tile_footprint(tanzania_example_image):
    """Test a tile footprint recovery
    """
    ds = gdal.Open(str(tanzania_example_image))
    geofeatures = get_image_features(ds)
    min_x = min_y = 0
    tile_width = ds.RasterXSize
    tile_footprint = get_tile_footprint(
        geofeatures, min_x, min_y, tile_width
    )
    assert tile_footprint.is_valid
    tile_bounds = tile_footprint.bounds
    assert geofeatures["north"] in tile_bounds
    assert geofeatures["south"] in tile_bounds
    assert geofeatures["east"] in tile_bounds
    assert geofeatures["west"] in tile_bounds


def test_rectangle_tile_footprint(tanzania_example_image):
    """Test a tile footprint recovery, based on the reference test image (see
    'tests/data/tanzania/input/training/')

    The full image is considered as the tile, its bounds must equal the image
    coordinates
    """
    ds = gdal.Open(str(tanzania_example_image))
    geofeatures = get_image_features(ds)
    min_x = min_y = 0
    tile_width = ds.RasterXSize
    tile_height = int(ds.RasterYSize / 2)
    tile_footprint = get_tile_footprint(
        geofeatures, min_x, min_y, tile_width, tile_height
    )
    assert tile_footprint.is_valid
    tile_bounds = tile_footprint.bounds
    tile_south = (geofeatures["south"]
                  + (geofeatures["north"] - geofeatures["south"]) / 2
                  )
    assert tile_south in tile_bounds
    assert geofeatures["north"] in tile_bounds
    assert geofeatures["east"] in tile_bounds
    assert geofeatures["west"] in tile_bounds


def test_extract_empty_tile_items(
        tanzania_example_image, tanzania_example_labels
):
    """Test the extraction of polygons that overlap a given squared tile, based
    on a reference test image (see 'tests/data/tanzania/input/training/')

    The tests is focused on an empty tile, that must provide an empty item set
    """
    ds = gdal.Open(str(tanzania_example_image))
    geofeatures = get_image_features(ds)
    labels = gpd.read_file(tanzania_example_labels)
    labels = labels.loc[~labels.geometry.isna(), ["condition", "geometry"]]
    none_mask = [lc is None for lc in labels.condition]
    labels.loc[none_mask, "condition"] = "Complete"
    empty_tile_items = extract_tile_items(
        geofeatures, labels, 450, 450, 100, 100
    )
    assert empty_tile_items.shape[0] == 0


def test_extract_tile_items(tanzania_example_image, tanzania_example_labels):
    """Test the extraction of polygons that overlap a given squared tile, based
    on a reference test image (see 'tests/data/tanzania/input/training/')

    The tests check that:
    - the example image contains 7 valid items
    - the items are 'Polygon' (in opposition to 'MultiPolygon')
    - the item union is contained into the tile footprint (overlapping items
    are cutted out so as out-of-image parts are removed)
    """
    ds = gdal.Open(str(tanzania_example_image))
    geofeatures = get_image_features(ds)
    labels = gpd.read_file(tanzania_example_labels)
    labels = labels.loc[~labels.geometry.isna(), ["condition", "geometry"]]
    none_mask = [lc is None for lc in labels.condition]
    labels.loc[none_mask, "condition"] = "Complete"
    tile_items = extract_tile_items(
        geofeatures, labels, 0, 0, 1000, 1000
    )
    expected_items = 7
    assert tile_items.shape[0] == expected_items
    assert np.all([
        geom.is_valid for geom in tile_items["geometry"]
    ])
    assert np.all([
        geom.geom_type == "Polygon" for geom in tile_items["geometry"]
    ])
    item_bounds = tile_items.unary_union.bounds
    assert (item_bounds[0] >= geofeatures["west"]
            and item_bounds[0] <= geofeatures["east"])
    assert (item_bounds[1] >= geofeatures["south"]
            and item_bounds[1] <= geofeatures["north"])
    assert (item_bounds[2] >= geofeatures["west"]
            and item_bounds[2] <= geofeatures["east"])
    assert (item_bounds[3] >= geofeatures["south"]
            and item_bounds[3] <= geofeatures["north"])
