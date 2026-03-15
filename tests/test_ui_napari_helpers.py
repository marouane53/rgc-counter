import numpy as np
import pytest

from src.ui_napari.helpers import (
    format_xy_text,
    landmarks_from_points,
    parse_optional_float,
    parse_optional_int,
    parse_xy_text,
)


def test_parse_xy_text_round_trip():
    xy = parse_xy_text("12.5, 7")
    assert xy == (12.5, 7.0)
    assert format_xy_text(xy) == "12.5, 7.0"


def test_optional_numeric_parsers_accept_blank():
    assert parse_optional_float("") is None
    assert parse_optional_int("") is None
    assert parse_optional_float("3.25") == 3.25
    assert parse_optional_int("7") == 7


def test_landmarks_from_points_uses_yx_input_order():
    points = np.array([[5.0, 10.0], [2.0, 12.0]])
    landmarks = landmarks_from_points(points)
    assert landmarks["onh_xy"] == (10.0, 5.0)
    assert landmarks["dorsal_xy"] == (12.0, 2.0)


def test_landmarks_from_points_requires_two_points():
    with pytest.raises(ValueError):
        landmarks_from_points(np.array([[5.0, 10.0]]))
