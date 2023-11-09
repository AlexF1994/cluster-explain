from typing import List, Optional, Type

import pandas as pd
import numpy as np
from nptyping import NDArray, Shape
from nptyping.typing_ import Floating, Int

from cxplain.base_explainer import BaseExplainer, ExplainedClustering
from cxplain.metrics import EuclideanMetric, Metric


class GradientExplainer(BaseExplainer):
    """
    Explain clustering results using pointwise gradient of the cluster dissimilarity measure.

    Attributes:
        data (NDArray[Shape["* num_obs, * num_features"], Floating]): Input data for clustering.
        cluster_centers (NDArray[Shape["* num_clusters, * num_features"], Floating]): Cluster centers
                                                                                      for the input data.
        cluster_predictions (NDArray[Shape["* num_obs"], Int]): Cluster predictions for the input data.
        metric (Optional[Type[Metric]]): Metric class for distance and gradient calculations
                                         (default is EuclideanMetric).
        enable_abs_calculation (bool): Flag to enable/disable absolute value calculation for relevance scores.
        feature_names (Optional[List[str]]): Optional list of feature names.
        num_features (int): The number of features in the input data.

    Methods:
        - fit(self):
            Fits the explainer.

        - _calculate_pointwise_relevance(self) -> pd.DataFrame:
            Computes pointwise feature relevance scores based on the pointwise gradient of cluster loss.

        - _calculate_cluster_relevance(self, pointwise_scores) -> pd.DataFrame:
            Computes cluster-wise feature relevance scores based on pointwise scores.

        - _calc_abs_value(self, df: pd.DataFrame, enabled) -> pd.DataFrame:
            Calculates absolute values for a DataFrame if enabled, or returns the original DataFrame.

        - _calculate_global_relevance(self, pointwise_scores) -> pd.Series:
            Computes global feature relevance scores based on pointwise scores.

        - explain(self) -> ExplainedClustering:
            Explains clustering results by computing pointwise, cluster, and global feature relevance scores.

    Example:

    >>> # Create a GradientExplainer instance
    >>> data = ...  # Input data for clustering
    >>> cluster_centers = ...  # Cluster centers for the input data
    >>> cluster_predictions = ...  # Cluster predictions for the input data
    >>> feature_names = ...  # Optional list of feature names
    >>> explainer = GradientExplainer(data, cluster_centers, cluster_predictions, metric=EuclideanMetric,
    ...                               enable_abs_calculation=True, feature_names=feature_names)
    >>> # Fit the explainer
    >>> explainer.fit()
    >>> # Explain clustering results
    >>> explained_result = explainer.explain()
    """

    def __init__(
        self,
        data: NDArray[Shape["* num_obs, * num_features"], Floating],  # type: ignore
        cluster_centers: NDArray[Shape["* num_clusters, * num_features"], Floating],  # type: ignore
        cluster_predictions: NDArray[Shape["* num_obs"], Int],  # type: ignore
        metric: Optional[Type[Metric]] = None,
        enable_abs_calculation: bool = True,
        feature_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.data = data
        self.cluster_centers = cluster_centers
        self.cluster_predictions = cluster_predictions
        self.metric = metric if metric else EuclideanMetric
        self.num_features = self.data.shape[1]
        self.enable_abs_calculation = enable_abs_calculation
        self.feature_names = feature_names
        self.is_fitted = True

    def fit(self):
        """
        Fits the explainer, making it ready for use.

        Example:

        >>> # Fit the explainer
        >>> explainer.fit()
        """
        return self

    def _calculate_pointwise_relevance(self) -> pd.DataFrame:
        """
        Computes pointwise feature relevance scores based on the pointwise gradient of cluster loss.

        Returns:
            pd.DataFrame: Pointwise feature relevance scores.

        Example:

        >>> # Compute pointwise feature relevance scores
        >>> pointwise_relevance = explainer._calculate_pointwise_relevance()
        """
        self._check_fitted()
        relevant_cluster_centers = np.take(
            self.cluster_centers, self.cluster_predictions, axis=0
        )
        gradient_values = self.metric.calculate_gradient(
            self.data, relevant_cluster_centers
        )
        return pd.DataFrame(gradient_values).pipe(
            self._rename_feature_columns, self.num_features, self.feature_names
        )

    def _calculate_cluster_relevance(
        self, pointwise_scores: pd.DataFrame
    ) -> pd.DataFrame:  # TODO: duplicated code
        """
        Computes cluster-wise feature relevance scores based on pointwise scores.

        Args:
            pointwise_scores: Pointwise feature relevance scores.

        Returns:
            pd.DataFrame: Cluster-wise feature relevance scores.

        Example:

        >>> # Compute cluster-wise feature relevance scores
        >>> cluster_relevance = explainer._calculate_cluster_relevance(pointwise_scores)
        """
        self._check_fitted()
        return (
            pointwise_scores.assign(assigned_clusters=self.cluster_predictions)
            .pipe(self._calc_abs_value, self.enable_abs_calculation)
            .groupby(["assigned_clusters"])
            .mean()
        )

    def _calc_abs_value(self, df: pd.DataFrame, enabled: bool) -> pd.DataFrame:
        """
        Calculates absolute values for a DataFrame if enabled, or returns the original DataFrame.

        Args:
            df: The input DataFrame.
            enabled: Flag to enable/disable absolute value calculation.

        Returns:
            pd.DataFrame: The original or absolute value DataFrame.

        Example:

        >>> # Calculate absolute values
        >>> df_abs = explainer._calc_abs_value(df, enabled=True)
        """
        return df.abs() if enabled else df.copy()

    def _calculate_global_relevance(
        self, pointwise_scores: pd.DataFrame
    ) -> pd.Series:  # TODO: duplicated code
        """
        Computes global feature relevance scores based on pointwise scores.

        Args:
            pointwise_scores: Pointwise feature relevance scores.

        Returns:
            pd.Series: Global feature relevance scores.

        Example:

        >>> # Compute global feature relevance scores
        >>> global_relevance = explainer._calculate_global_relevance(pointwise_scores)
        """
        return pointwise_scores.pipe(
            self._calc_abs_value, self.enable_abs_calculation
        ).mean()

    def explain(self) -> ExplainedClustering:  # TODO duplicated Code
        """
        Explains clustering results by computing pointwise, cluster, and global feature relevance scores.

        Returns:
            ExplainedClustering: An instance of ExplainedClustering containing feature relevance scores.

        Example:

        >>> # Explain clustering results
        >>> explained_result = explainer.explain()
        """
        self._check_fitted()
        pointwise_relevance = self._calculate_pointwise_relevance()
        cluster_relevance = self._calculate_cluster_relevance(
            pointwise_scores=pointwise_relevance
        )
        global_relevance = self._calculate_global_relevance(
            pointwise_scores=pointwise_relevance
        )

        return ExplainedClustering(
            pointwise_relevance=pointwise_relevance,
            cluster_relevance=cluster_relevance,
            global_relevance=global_relevance,
        )
