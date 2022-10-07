from abc import ABC, abstractmethod
from typing import Type

from nptyping import NDArray

from cxplain.errors import MetricNotImplementedError


class Metric(ABC):
    
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

class ManhattenMetric(Metric):
    @staticmethod
    def calculate(x: NDArray, y: NDArray) -> NDArray: 
        """
        Calculate pointwise Manhatten distance between x and y. 

        Args:
            x (NDArray): Input to distance calculation.
            y (NDArray): Input to distance calculation.

        Returns:
            NDArray: Pointwise Manhatten distance.
        """
        return abs(x - y)

class EuclideanMetric(Metric):
    @staticmethod
    def calculate(x: NDArray, y: NDArray) -> NDArray:
        """
        Calculate pointwise Euclidean distance between x and y. 

        Args:
            x (NDArray): Input to distance calculation.
            y (NDArray): Input to distance calculation.

        Returns:
            NDArray: Pointwise Euclidean distance. 
        """
        return (x - y)**2
        

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
        return ManhattenMetric
    
    if metric_name == "euclidean":
        return EuclideanMetric
    
    else:
        raise MetricNotImplementedError(f"Your requested metric {metric_name} is not yet implemented!")


