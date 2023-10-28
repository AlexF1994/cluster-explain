from typing import List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from turtle import Shape

import pandas as pd
from nptyping import NDArray, Shape  # noqa: F811
from nptyping.typing_ import Floating, Int
from ExKMC.Tree import Tree
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted

from cxplain.base_explainer import BaseExplainer, ExplainedClustering


class DecisionTreeExplainer(BaseExplainer):
    def __init__(
        self,
        data: NDArray[Shape["* num_obs, * num_features"], Floating],  # type: ignore
        cluster_predictions: NDArray[Shape["* num_obs"], Int],  # type: ignore
        feature_names: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__()
        self.data = data
        self.cluster_predictions = cluster_predictions
        self.tree = DecisionTreeClassifier(**kwargs)
        self.num_features = (self.data.shape[1],)
        self.feature_names = feature_names

    def fit(self):
        self.tree.fit(self.data, self.cluster_predictions)
        self.is_fitted = True
        return self

    def _calculate_global_relevance(self) -> pd.DataFrame:
        feature_importances = self.tree.tree_.compute_feature_importances(
            normalize=False
        )
        return pd.DataFrame(feature_importances).T.pipe(
            self._rename_feature_columns, self.num_features, self.feature_names
        )

    def explain(self) -> ExplainedClustering:
        self._check_fitted()
        feature_importances = self._calculate_global_relevance()
        return ExplainedClustering(global_relevance=feature_importances.squeeze())


class RandomForestExplainer(BaseExplainer):
    def __init__(
        self,
        data: NDArray[Shape["* num_obs, * num_features"], Floating],  # type: ignore
        cluster_predictions: NDArray[Shape["* num_obs"], Int],  # type: ignore
        feature_names: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__()
        self.data = data
        self.cluster_predictions = cluster_predictions
        self.forest = RandomForestClassifier(**kwargs)
        self.num_features = self.data.shape[1]
        self.feature_names = feature_names

    def fit(self):
        self.forest.fit(self.data, self.cluster_predictions)
        self.is_fitted = True
        return self

    def _calculate_global_relevance(self) -> pd.DataFrame:
        feature_importances = self.forest.feature_importances_
        return pd.DataFrame(feature_importances).T.pipe(
            self._rename_feature_columns, self.num_features, self.feature_names
        )

    def explain(self) -> ExplainedClustering:
        self._check_fitted()
        feature_importances = self._calculate_global_relevance()
        return ExplainedClustering(global_relevance=feature_importances.squeeze())


class ExKMCExplainer(BaseExplainer):
    def __init__(
        self,
        data: NDArray[Shape["* num_obs, * num_features"], Floating],  # type: ignore
        kmeans_fitted: KMeans,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__()
        self.data = data
        self.kmeans = kmeans_fitted
        self.tree = Tree(**kwargs)
        self.num_features = self.data.shape[1]
        self.feature_names = feature_names

    def fit(self):
        check_is_fitted(self.kmeans)
        self.tree.fit(self.data, self.kmeans)
        self.is_fitted = True
        return self

    def _calculate_global_relevance(self) -> pd.DataFrame:
        feature_importances = self.tree._feature_importance
        return pd.DataFrame(feature_importances).T.pipe(
            self._rename_feature_columns, self.num_features, self.feature_names
        )

    def explain(self) -> ExplainedClustering:
        self._check_fitted()
        feature_importances = self._calculate_global_relevance()
        return ExplainedClustering(global_relevance=feature_importances.squeeze())
