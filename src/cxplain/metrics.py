from abc import ABC, abstractmethod
from typing import Type

from nptyping import NDArray

from cxplain.errors import MetricNotImplementedError

## TODO: evaluate if numpy functionality for metric calculation is better


class Metric(ABC):
    """
    Base class for metrics used by different explainer.
    """

    @staticmethod
    @abstractmethod
    def calculate(x: NDArray, y: NDArray):
        """
        Abstract method to caclculate the distance between x and y.

        Args:
            x (NDArray): Input to distance calculation.
            y (NDArray): Input to distance calculation.
        """
        pass

    @staticmethod
    @abstractmethod
    def calculate_gradient(x: NDArray, y: NDArray):
        """
        Abstract method to caclculate the gradient of the metric.

        Args:
            x (NDArray): Input to metric calculation.
            y (NDArray): Input to metric calculation.
        """
        pass


class ManhattanMetric(Metric):
    """
    A class for calculating pointwise Manhattan distance and its gradient between two NumPy arrays.
    """

    @staticmethod
    def calculate(x: NDArray, y: NDArray) -> NDArray:
        """
        Calculate pointwise Manhattan distance between x and y.

        Args:
            x (NDArray): Input to distance calculation.
            y (NDArray): Input to distance calculation.

        Returns:
            NDArray: Pointwise Manhattan distance.

        Example:
            .. code-block:: python

                import numpy as np

                x = np.array([1, 2, 3])
                y = np.array([4, 5, 6])
                result = ManhattenMetric.calculate(x, y)
                print(result)  # Output: array([3, 3, 3])

        """
        return abs(x - y)

    @staticmethod
    def calculate_gradient(x: NDArray, y: NDArray) -> NDArray:
        """
        Calculate the pointwise gradient between x and y.

        The pointwise gradient is computed as (x - y) / |x - y|.

        Args:
            x (NDArray): Input for gradient calculation.
            y (NDArray): Input for gradient calculation.

        Returns:
            NDArray: Pointwise gradient.

        Example:
            .. code-block:: python

                import numpy as np

                x = np.array([1.0, 2.0, 3.0])
                y = np.array([4.0, 5.0, 6.0])
                gradient = ManhattenMetric.calculate_gradient(x, y)
                print(gradient)  # Output: array([-1., -1., -1.])

        """
        return (x - y) / abs(x - y)


class EuclideanMetric(Metric):
    """
    A class for calculating pointwise Euclidean distance and its gradient between two NumPy arrays.
    """

    @staticmethod
    def calculate(x: NDArray, y: NDArray) -> NDArray:
        """
        Calculate pointwise Euclidean distance between x and y.

        Args:
            x (NDArray): Input to distance calculation.
            y (NDArray): Input to distance calculation.

        Returns:
            NDArray: Pointwise Euclidean distance.

        Example:
            .. code-block:: python

                import numpy as np

                x = np.array([1.0, 2.0, 3.0])
                y = np.array([4.0, 5.0, 6.0])

                # Calculate pointwise Euclidean distance
                distance = EuclideanMetric.calculate(x, y)
                print(distance)  # Output: array([9., 9., 9.])

        """
        return (x - y) ** 2

    @staticmethod
    def calculate_gradient(x: NDArray, y: NDArray) -> NDArray:
        """
        Calculate the gradient of pointwise Euclidean distance between x and y.

        Args:
            x (NDArray): Input for gradient calculation.
            y (NDArray): Input for gradient calculation.

        Returns:
            NDArray: Pointwise Euclidean gradient.

        Example:
            .. code-block:: python

                import numpy as np

                x = np.array([1.0, 2.0, 3.0])
                y = np.array([4.0, 5.0, 6.0])

                # Calculate pointwise Euclidean gradient
                gradient = EuclideanMetric.calculate_gradient(x, y)
                print(gradient)  # Output: array([-6., -6., -6.])

        """
        return 2 * (x - y)


def get_distance_metric(metric_name: str) -> Type[Metric]:
    """
    Factory method for distance metrices.

    Args:
        metric_name (str): Name of the distance metric that should be applied.

    Raises:
        MetricNotImplementedError: Raised if desired metric is not yet implemented.

    Returns:
        Type[Metric]: The desired distance metric.
    """
    if metric_name == "manhattan":
        return ManhattanMetric()

    if metric_name == "euclidean":
        return EuclideanMetric()

    else:
        raise MetricNotImplementedError(
            f"Your requested metric {metric_name} is not yet implemented!"
        )
