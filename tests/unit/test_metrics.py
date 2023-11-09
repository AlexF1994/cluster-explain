import numpy as np
import pytest

from cxplain.metrics import (
    EuclideanMetric,
    ManhattenMetric,
    MetricNotImplementedError,
    get_distance_metric,
)


@pytest.fixture()
def x():
    return np.array([[1, 1], [2, 2]])


@pytest.fixture()
def y():
    return np.array([[2, 2], [1.5, 2.5]])


def test_get_distance_metric():
    assert isinstance(get_distance_metric("manhatten"), ManhattenMetric)
    assert isinstance(get_distance_metric("euclidean"), EuclideanMetric)
    with pytest.raises(MetricNotImplementedError):
        get_distance_metric("bla")


def test_calculate_manhatten(x, y):
    metric = get_distance_metric("manhatten")
    expected = np.array([[1, 1], [0.5, 0.5]])
    actual = metric.calculate(x, y)
    np.testing.assert_allclose(expected, actual)  # type: ignore


def test_calculate_euclidean(x, y):
    metric = get_distance_metric("euclidean")
    expected = np.array([[1, 1], [0.25, 0.25]])
    actual = metric.calculate(x, y)
    np.testing.assert_allclose(expected, actual)  # type: ignore
