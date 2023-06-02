from typing import List, Optional, Type

import pandas as pd
import numpy as np
from nptyping import NDArray, Shape
from nptyping.typing_ import Floating, Int

from cxplain.base_explainer import BaseExplainer, ExplainedClustering
from cxplain.metrics import EuclideanMetric, Metric


class GradientExplainer(BaseExplainer):
    """Explain clustering by gradient of cluster loss."""

    def __init__(
        self,
        data: NDArray[Shape["* num_obs, * num_features"], Floating],
        cluster_centers: NDArray[Shape["* num_clusters, * num_features"], Floating],
        cluster_predictions: NDArray[Shape["* num_obs"], Int],
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
        return self

    def _calculate_pointwise_relevance(self) -> pd.DataFrame:
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
        self._check_fitted()
        return (
            pointwise_scores.assign(assigned_clusters=self.cluster_predictions)
            .pipe(self._calc_abs_value, self.enable_abs_calculation)
            .groupby(["assigned_clusters"])
            .mean()
        )

    def _calc_abs_value(self, df: pd.DataFrame, enabled: bool) -> pd.DataFrame:
        return df.abs() if enabled else df.copy()

    def _calculate_global_relevance(
        self, pointwise_scores: pd.DataFrame
    ) -> pd.Series:  # TODO: duplicated code
        return pointwise_scores.pipe(
            self._calc_abs_value, self.enable_abs_calculation
        ).mean()

    def explain(self) -> ExplainedClustering:  # TODO duplicated Code
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
