"""Unit tests that address geometries.py module functions
"""

import numpy as np
import pytest

from deeposlandia.datasets.tanzania import (
    get_x_pixel, get_y_pixel, get_x_geocoord, get_y_geocoord
)


def test_get_pixel():
    """Test the transformation from georeferenced coordinates to pixel
    """
    geocoords = [10000.0, 15000.0, 20000.0]
    min_coord, max_coord = 0, 30000.0
    size_in_pixel = 500
    true_pixel_coord = [166, 250, 333]
    pixel_coord = get_x_pixel(
        geocoords, min_coord, max_coord, size_in_pixel
    )
    assert np.all(pixel_coord == true_pixel_coord)
    single_geocoord = geocoords[0]
    single_pixel_coord = get_x_pixel(
        single_geocoord, min_coord, max_coord, size_in_pixel
    )
    assert single_pixel_coord == true_pixel_coord[0]
    str_geocoord = "15000"
    with pytest.raises(TypeError):
        get_x_pixel(str_geocoord, min_coord, max_coord, size_in_pixel)


def test_get_geocoord():
    """Test the transformation from pixel to georeferenced coordinates
    """
    pixels = [166, 250, 333]
    min_coord, max_coord = 0, 30000.0
    size_in_pixel = 500
    true_geocoord = [9960.0, 15000.0, 19980.0]
    geocoord = get_x_geocoord(pixels, min_coord, max_coord, size_in_pixel)
    assert np.all(geocoord == true_geocoord)
    single_pixel = pixels[0]
    single_geocoord = get_x_geocoord(
        single_pixel, min_coord, max_coord, size_in_pixel
    )
    assert single_geocoord == true_geocoord[0]
    str_pixel = "250"
    with pytest.raises(TypeError):
        get_x_geocoord(str_pixel, min_coord, max_coord, size_in_pixel)
