"""Unit tests that address geometries.py module functions
"""

import pytest

import geopandas as gpd
import numpy as np
from osgeo import gdal
from shapely.geometry import Polygon, MultiPolygon

from deeposlandia.geometries import (
    extract_points_from_polygon,
    extract_tile_items,
    get_geocoord,
    get_image_features,
    get_pixel,
    get_tile_footprint,
    extract_geometry_vertices,
    retrieve_area_color,
    vectorize_mask,
    rasterize_polygons,
    pixel_to_geocoord,
    convert_to_geocoord,
)


def test_get_pixel():
    """Test the transformation from georeferenced coordinates to pixel.

    The transformation conserves the coordinate data structure (scalar or
    list).
    """
    geocoords = [10000.0, 15000.0, 20000.0]
    min_coord, max_coord = 0, 30000.0
    size_in_pixel = 500
    true_pixel_coord = [166, 250, 333]
    pixel_coord = get_pixel(geocoords, min_coord, max_coord, size_in_pixel)
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
    """Test the transformation from pixel to georeferenced coordinates.

    The transformation conserves the coordinate data structure (scalar or
    list).
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
    """Test the image geographic feature recovering:
    - 'south', 'north', 'west' and 'east' are the image geographic coordinates,
    hence floating numbers
    - west is smaller than east
    - south is smaller than north
    - srid is an integer geocode
    - width and height are strictly positive int, as they represent the image
    size, in pixels
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
    """Test a polygon point extraction.

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
    polygon = Polygon(((x1, y1), (x1, y2), (x2, y1), (x1, y1)))
    points = extract_points_from_polygon(polygon, geofeatures, min_x, min_y)
    expected_points = np.array([[0, 0], [500, 0], [0, 250], [0, 0]])
    assert np.all(points == expected_points)


def test_square_tile_footprint(tanzania_example_image):
    """Test a tile footprint recovery, based on the reference test image (see
    'tests/data/tanzania/input/training/').

    The full image is considered as the tile, its bounds must equal the image
    coordinates.
    """
    ds = gdal.Open(str(tanzania_example_image))
    geofeatures = get_image_features(ds)
    min_x = min_y = 0
    tile_width = ds.RasterXSize
    tile_footprint = get_tile_footprint(geofeatures, min_x, min_y, tile_width)
    assert tile_footprint.is_valid
    tile_bounds = tile_footprint.bounds
    assert geofeatures["north"] in tile_bounds
    assert geofeatures["south"] in tile_bounds
    assert geofeatures["east"] in tile_bounds
    assert geofeatures["west"] in tile_bounds


def test_rectangle_tile_footprint(tanzania_example_image):
    """Test a tile footprint recovery, based on the reference test image (see
    'tests/data/tanzania/input/training/').

    The considered tile is the top-half of the image, its bounds must equal
    the image coordinates, except the south bound that must equal the mean
    between north and south coordinates.
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
    tile_south = (
        geofeatures["south"]
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
    on a reference test image (see 'tests/data/tanzania/input/training/').

    The tests is focused on an empty tile, that must provide an empty item set.
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
    on a reference test image (see 'tests/data/tanzania/input/training/').

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
    tile_items = extract_tile_items(geofeatures, labels, 0, 0, 1000, 1000)
    expected_items = 7
    assert tile_items.shape[0] == expected_items
    assert np.all([geom.is_valid for geom in tile_items["geometry"]])
    assert np.all(
        [geom.geom_type == "Polygon" for geom in tile_items["geometry"]]
    )
    item_bounds = tile_items.unary_union.bounds
    assert (
        item_bounds[0] >= geofeatures["west"]
        and item_bounds[0] <= geofeatures["east"]
    )
    assert (
        item_bounds[1] >= geofeatures["south"]
        and item_bounds[1] <= geofeatures["north"]
    )
    assert (
        item_bounds[2] >= geofeatures["west"]
        and item_bounds[2] <= geofeatures["east"]
    )
    assert (
        item_bounds[3] >= geofeatures["south"]
        and item_bounds[3] <= geofeatures["north"]
    )


def test_extract_geometry_vertices(tanzania_raw_image_size):
    """Test the polygon vertice extraction from a raster mask

    Test a simple case with two expected polygons. The considered function uses
    OpenCV library to find polygon contours, and some approximated methods are
    implied. Hence exact polygon vertice coordinates are not tested.

    Two variant are tested, regarding the "structure" parameter, fundamental in
    OpenCV operations. If it is too large compared to polygon sizes, they are
    not detected.
    """
    mask = np.zeros([10, 10], dtype=np.uint8)
    x1, y1 = 1, 2
    x2, y2 = 3, 4
    x3, y3 = 3, 6
    x4, y4 = 7, 9
    mask[y1:y2, x1:x2] = 1
    mask[y3:y4, x3:x4] = 1
    undetected_vertices, _ = extract_geometry_vertices(
        mask, structure_size=(10, 10)
    )
    assert len(undetected_vertices) == 0
    bigger_mask = np.zeros(
        [tanzania_raw_image_size, tanzania_raw_image_size], dtype=np.uint8
    )
    x1, y1 = 100, 200
    x2, y2 = 300, 400
    x3, y3 = 300, 600
    x4, y4 = 700, 900
    bigger_mask[y1:y2, x1:x2] = 1
    bigger_mask[y3:y4, x3:x4] = 1
    polygon_vertices, _ = extract_geometry_vertices(
        bigger_mask, structure_size=(10, 10)
    )
    assert len(polygon_vertices) == 2


def test_retrieve_area_color(tanzania_raw_image_size):
    """Test the label retrieving function, in order to assign a label to a
    given area thanks to its color

    One tests three cases:
      - an empty area must return "0" label
      - an area filled with a color must return corresponding label
      - an area partially filled must return the most encountered label
    """
    label_dicts = [
        {"id": 0, "color": [0, 0, 0]},
        {"id": 1, "color": [50, 200, 50]}
    ]
    data = np.zeros(
        [tanzania_raw_image_size, tanzania_raw_image_size, 3], dtype=np.uint8
    )
    x1, y1 = 100, 200
    x2, y2 = 300, 400
    contour = np.array(
        [[[x1, y1]], [[x1, y2]], [[x2, y2]], [[x2, y1]]],
    )
    assert retrieve_area_color(data, contour, label_dicts) == 0
    data[y1:y2, x1:x2] = label_dicts[1]["color"]
    assert retrieve_area_color(data, contour, label_dicts) == 1
    x2 += 100
    contour = np.array(
        [[[x1, y1]], [[x1, y2]], [[x2, y2]], [[x2, y1]]],
    )
    assert retrieve_area_color(data, contour, label_dicts) == 1


def test_vectorize_mask(tanzania_raw_image_size):
    """Test the mask vectorization operation, that transform raster mask into a
    MultiPolygon.

    Test a simple case with two expected polygons. The considered function uses
    OpenCV library to find polygon contours, and some approximated methods are
    implied. Hence exact polygon vertice coordinates are not tested.
    """
    label_dicts = [
        {"id": 1, "color": [50, 200, 50]},
        {"id": 2, "color": [200, 50, 50]}
    ]
    mask = np.zeros(
        [tanzania_raw_image_size, tanzania_raw_image_size], dtype=np.uint8
    )
    data = np.zeros(
        [tanzania_raw_image_size, tanzania_raw_image_size, 3], dtype=np.uint8
    )
    empty_labels, empty_multipolygon = vectorize_mask(mask, data, label_dicts)
    assert len(empty_labels) == 0
    assert len(empty_multipolygon) == 0
    x1, y1 = 100, 200
    x2, y2 = 300, 400
    x3, y3 = 300, 600
    x4, y4 = 700, 900
    mask[y1:y2, x1:x2] = label_dicts[0]["id"]
    mask[y3:y4, x3:x4] = label_dicts[1]["id"]
    data[y1:y2, x1:x2] = label_dicts[0]["color"]
    data[y3:y4, x3:x4] = label_dicts[1]["color"]
    labels, multipolygon = vectorize_mask(mask, data, label_dicts)
    assert len(labels) == 2
    assert np.sum(labels == 1) == 1
    assert np.sum(labels == 2) == 1
    assert len(multipolygon) == 2


def test_rasterize_polygons(tanzania_raw_image_size):
    """Test the rasterization process

    Considering the "no-polygon" case, the function must return an empty mask.

    Considering a polygon that fill the left part of the original image. The
    rasterized mask must be filled with "1" on this part, and with "0" on the
    right part.
    """
    mask = rasterize_polygons(
        [],
        np.array([]),
        tanzania_raw_image_size,
        tanzania_raw_image_size
    )
    assert mask.shape == (tanzania_raw_image_size, tanzania_raw_image_size)
    assert np.unique(mask) == np.array([0])
    x1 = int(tanzania_raw_image_size / 3)
    x2 = int(2 * tanzania_raw_image_size / 3)
    polygon1 = Polygon(
        shell=(
            (0, 0),
            (x1, 0),
            (x1, tanzania_raw_image_size),
            (0, tanzania_raw_image_size),
            (0, 0),
        )
    )
    polygon2 = Polygon(
        shell=(
            (x1, 0),
            (x2, 0),
            (x2, tanzania_raw_image_size),
            (x1, tanzania_raw_image_size),
            (x1, 0),
        )
    )
    labels = [1, 2]
    mask = rasterize_polygons(
        MultiPolygon([polygon1, polygon2]),
        np.array(labels),
        tanzania_raw_image_size,
        tanzania_raw_image_size,
    )
    mask_polygon_1 = mask[:tanzania_raw_image_size, :x1]
    assert np.unique(mask_polygon_1) == labels[0]
    mask_polygon_2 = mask[:tanzania_raw_image_size, (1 + x1):x2]
    assert np.unique(mask_polygon_2) == labels[1]
    mask_no_polygon = mask[:tanzania_raw_image_size, (1 + x2):]
    assert np.unique(mask_no_polygon) == 0


def test_pixel_to_geocoord(tanzania_example_image, tanzania_raw_image_size):
    """Test the transformation of a Polygon from pixel to georeferenced
    coordinates

    Use the full image footprint as a reference polygon.
    """
    ds = gdal.Open(str(tanzania_example_image))
    geofeatures = get_image_features(ds)
    polygon = Polygon(
        shell=(
            (0, 0),
            (tanzania_raw_image_size, 0),
            (tanzania_raw_image_size, tanzania_raw_image_size),
            (0, tanzania_raw_image_size),
            (0, 0),
        )
    )
    expected_points = np.array(
        [
            [geofeatures["west"], geofeatures["north"]],
            [geofeatures["east"], geofeatures["north"]],
            [geofeatures["east"], geofeatures["south"]],
            [geofeatures["west"], geofeatures["south"]],
            [geofeatures["west"], geofeatures["north"]],
        ]
    )
    points = pixel_to_geocoord(polygon.exterior, geofeatures)
    assert np.all(points == expected_points)


def test_convert_to_geocoord(tanzania_example_image, tanzania_raw_image_size):
    """Test the convertion of a set of pixel-referenced polygons to
    georeferenced ones.

    Some of the polygon may include holes (hence interior points). We test the
    following design, where there are two polygons, of whom one has a hole:
        ____
       |1110|
       |1010|
       |1110|
       |0002|
        ----
    """
    x0 = y0 = 0
    x1 = y1 = int(tanzania_raw_image_size / 4)
    x2 = y2 = int(tanzania_raw_image_size / 2)
    x3 = y3 = int(tanzania_raw_image_size * 3 / 4)
    x4 = y4 = tanzania_raw_image_size
    polygon1 = Polygon(
        shell=((x0, y0), (x3, y0), (x3, y3), (x0, y3), (x0, y0)),
        holes=[((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1))],
    )
    polygon2 = Polygon(
        shell=((x3, y3), (x4, y3), (x4, y4), (x3, y4), (x3, y3))
    )
    multipolygon = MultiPolygon([polygon1, polygon2])
    ds = gdal.Open(str(tanzania_example_image))
    geofeatures = get_image_features(ds)
    converted_multipolygon = convert_to_geocoord(multipolygon, geofeatures)
    expected_x = [
        (geofeatures["west"] + (geofeatures["east"] - geofeatures["west"]) * i)
        for i in np.linspace(0, 1, 5)
    ]
    expected_y = [
        (
            geofeatures["north"]
            + (geofeatures["south"] - geofeatures["north"]) * i
        )
        for i in np.linspace(0, 1, 5)
    ]
    expected_polygon1 = Polygon(
        shell=(
            (expected_x[0], expected_y[0]),
            (expected_x[3], expected_y[0]),
            (expected_x[3], expected_y[3]),
            (expected_x[0], expected_y[3]),
            (expected_x[0], expected_y[0]),
        ),
        holes=[
            (
                (expected_x[1], expected_y[1]),
                (expected_x[2], expected_y[1]),
                (expected_x[2], expected_y[2]),
                (expected_x[1], expected_y[2]),
                (expected_x[1], expected_y[1]),
            )
        ],
    )
    expected_polygon2 = Polygon(
        shell=(
            (expected_x[3], expected_y[3]),
            (expected_x[4], expected_y[3]),
            (expected_x[4], expected_y[4]),
            (expected_x[3], expected_y[4]),
            (expected_x[3], expected_y[3]),
        )
    )
    assert converted_multipolygon[0] == expected_polygon1
    assert converted_multipolygon[1] == expected_polygon2
