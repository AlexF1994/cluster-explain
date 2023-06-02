from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from cxplain.errors import (
    InconsistentNamingError,
    NonExistingRelevanceError,
    NotFittedError,
)

# TODO remove code duplication in plotting of relevance scores
# TODO support for different data types (categorical)

index_type = Optional[Union[str, List[str], List[int], pd.Index]]


def color_mapping(value: float):
    if value >= 0:
        return "green"
    else:
        return "red"


@dataclass(frozen=True)
class GlobalExplainedClustering:
    global_relevance: pd.Series

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GlobalExplainedClustering):
            return False
        global_equal = np.all(
            np.isclose(self.global_relevance.values, other.global_relevance.values)
        )
        return global_equal

    def show_global_relevance(self):
        global_df = pd.DataFrame(self.global_relevance).reset_index(drop=False)
        sns.barplot(
            x=0,
            y="index",
            data=global_df,
            orient="h",
            palette=[color_mapping(value) for value in self.global_relevance.values],
        )
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
        plt.title("Global feature importance")
        plt.show()


@dataclass(frozen=True)
class ClusterExplainedClustering:
    cluster_relevance: pd.DataFrame

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClusterExplainedClustering):
            return False
        cluster_equal = np.all(
            np.isclose(self.cluster_relevance.values, other.cluster_relevance.values)
        )
        return cluster_equal

    def show_cluster_relevance(self, subset_index: index_type = None):
        relevances_to_plot = (
            self.cluster_relevance[subset_index]
            if subset_index
            else self.cluster_relevance
        )
        sns.heatmap(relevances_to_plot)
        plt.title("Clusterwise feature importance scores")
        plt.xlabel("Feature")
        plt.ylabel("Cluster")
        plt.show()

    def show_single_feature_relevance(
        self, feature: str, subset_index: index_type = None
    ):
        feature_importance = (
            self.cluster_relevance.loc[subset_index, [feature]]
            if subset_index
            else self.cluster_relevance[[feature]]
        )
        sns.barplot(
            x=feature,
            y="assigned_clusters",
            data=feature_importance.reset_index(drop=False),
            orient="h",
            palette=[color_mapping(value) for value in feature_importance.values],
        )
        plt.xlabel("Feature importance")
        plt.ylabel("Cluster")
        plt.title(f"Importance for feature: {feature}")
        plt.show()

    def show_single_cluster_relevance(self, cluster_index: int):
        observation_importance = self.cluster_relevance.loc[[cluster_index], :]
        sns.barplot(
            x=1,
            y="index",
            data=observation_importance.T.reset_index(drop=False),
            orient="h",
            palette=[
                color_mapping(value)
                for value in observation_importance.iloc[0, :].values
            ],
        )
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
        plt.title(f"Importance for Cluster: {cluster_index}")
        plt.show()


@dataclass(frozen=True)
class PointwiseExplainedClustering:
    pointwise_relevance: pd.DataFrame

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PointwiseExplainedClustering):
            return False
        pointwise_equal = np.all(
            np.isclose(
                self.pointwise_relevance.values, other.pointwise_relevance.values
            )
        )
        return pointwise_equal

    def show_pointwise_relevance(self, subset_index: index_type = None):
        relevances_to_plot = (
            self.pointwise_relevance.loc[subset_index, :]
            if subset_index
            else self.pointwise_relevance
        )
        sns.heatmap(relevances_to_plot)
        plt.title("Pointwise feature importance scores")
        plt.xlabel("Feature")
        plt.ylabel("Observation")
        plt.show()

    def show_single_feature_relevance(
        self, feature: str, subset_index: index_type = None
    ):
        feature_importance = (
            self.pointwise_relevance.loc[subset_index, [feature]]
            if subset_index
            else self.pointwise_relevance[[feature]]
        )
        sns.barplot(
            x=feature,
            y="index",
            data=feature_importance.reset_index(drop=False),
            orient="h",
            palette=[color_mapping(value) for value in feature_importance.values],
        )
        plt.xlabel("Feature importance")
        plt.ylabel("Observation")
        plt.title(f"Importance for feature: {feature}")
        plt.show()

    def show_single_observation_relevance(self, observation_index: int):
        observation_importance = self.pointwise_relevance.loc[observation_index, :]
        sns.barplot(
            x=1,
            y="index",
            data=observation_importance.T.reset_index(drop=False),
            orient="h",
            palette=[color_mapping(value) for value in observation_importance.values],
        )
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
        plt.title(f"Importance for observation: {observation_index}")
        plt.show()


class ExplainedClustering:
    def __init__(
        self,
        global_relevance: pd.Series,
        pointwise_relevance: Optional[pd.DataFrame] = None,
        cluster_relevance: Optional[pd.DataFrame] = None,
    ):
        self._pointwise_relevance = (
            PointwiseExplainedClustering(pointwise_relevance)
            if isinstance(pointwise_relevance, pd.DataFrame)
            else None
        )
        self._cluster_relevance = (
            ClusterExplainedClustering(cluster_relevance)
            if isinstance(cluster_relevance, pd.DataFrame)
            else None
        )
        self._global_relevance = GlobalExplainedClustering(global_relevance)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExplainedClustering):
            return False

        pointwise_equal = (
            (self._pointwise_relevance == other._pointwise_relevance)
            if self._pointwise_relevance
            else True
        )
        cluster_equal = (
            (self._cluster_relevance == other._cluster_relevance)
            if self._cluster_relevance
            else True
        )
        global_equal = self._global_relevance == other._global_relevance
        return pointwise_equal and cluster_equal and global_equal

    @property
    def pointwise_relevance(self) -> Optional[pd.DataFrame]:
        self._check_relevance_exists(self._pointwise_relevance)
        return self._pointwise_relevance.pointwise_relevance

    @property
    def cluster_relevance(self) -> Optional[pd.DataFrame]:
        self._check_relevance_exists(self._cluster_relevance)
        return self._cluster_relevance.cluster_relevance

    @property
    def global_relevance(self) -> pd.Series:
        return self._global_relevance.global_relevance

    @staticmethod
    def _check_relevance_exists(
        relevance: Optional[
            Union[PointwiseExplainedClustering, ClusterExplainedClustering]
        ] = None
    ):
        if not (
            isinstance(relevance, PointwiseExplainedClustering)
            or isinstance(relevance, ClusterExplainedClustering)
        ):
            raise NonExistingRelevanceError(
                "The specific relevance score doesn't exist for your explainer!"
            )

    def show_pointwise_relevance(self, subset_index: index_type = None):
        self._pointwise_relevance.show_pointwise_relevance(subset_index)

    def show_pointwise_relevance_for_feature(
        self, feature: str, subset_index: index_type = None
    ):
        self._pointwise_relevance.show_single_feature_relevance(feature, subset_index)

    def show_pointwise_relevance_for_observation(self, observation_index: int):
        self._pointwise_relevance.show_single_observation_relevance(observation_index)

    def show_cluster_relevance(self, subset_index: index_type = None):
        self._cluster_relevance.show_cluster_relevance(subset_index)

    def show_cluster_relevance_for_feature(
        self, feature: str, subset_index: index_type = None
    ):
        self._cluster_relevance.show_single_feature_relevance(feature, subset_index)

    def show_cluster_relevance_for_cluster(self, cluster_index: int):
        self._cluster_relevance.show_single_cluster_relevance(cluster_index)

    def show_global_relevance(self):
        self._global_relevance.show_global_relevance()


class BaseExplainer(ABC):
    """
    Base class for all cluster explainers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.is_fitted = False

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def explain(self) -> ExplainedClustering:
        pass

    def fit_explain(self) -> ExplainedClustering:
        self.fit()
        return self.explain()

    def _check_fitted(self):
        if not self.is_fitted:
            raise NotFittedError(
                "You have to calculate the feature wise distance matrix and the best alternative"
                " and the best alternative distances first!"
            )

    @staticmethod
    def _rename_feature_columns(
        df: pd.DataFrame, num_features: int, feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        if feature_names:
            if not num_features == len(feature_names):
                raise InconsistentNamingError
        return df.rename(
            {
                index_: feature_names[index_] if feature_names else f"R{index_ + 1}"
                for index_ in range(num_features)
            },
            axis=1,
        )
