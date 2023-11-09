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


@dataclass()
class GlobalExplainedClustering:
    """
    A class for representing global clustering explanations.

    Attributes:
        global_relevance (pd.Series): A Pandas Series containing global feature relevance scores.

    Methods:
        - __eq__(self, other) -> bool:
            Checks if two instances of GlobalExplainedClustering are equal.

        - show_global_relevance(self):
            Visualizes global feature relevance using a bar plot.

    Example:

    >>> import pandas as pd
    >>> # Create a GlobalExplainedClustering instance with global feature relevance
    >>> global_relevance = pd.Series([0.3, 0.5, 0.2], index=["feature_A", "feature_B", "feature_C"])
    >>> global_explanation = GlobalExplainedClustering(global_relevance)
    >>> # Check if two instances of GlobalExplainedClustering are equal
    >>> another_global_relevance = pd.Series([0.5, 0.5, 0.2], index=["feature_A", "feature_B", "feature_C"])
    >>> another_global_explanation = GlobalExplainedClustering(another_global_relevance)
    >>> global_explanation == another_global_explanation
    ... False
    >>> # Visualize global feature relevance
    >>> global_explanation.show_global_relevance()
    """

    global_relevance: pd.Series

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GlobalExplainedClustering):
            return False
        global_equal = np.all(
            np.isclose(self.global_relevance_df.values, other.global_relevance_df.values)  # type: ignore
        )
        return global_equal  # type: ignore

    def show_global_relevance(self):
        """
        Visualizes global feature relevance using a bar plot.

        Example:

        >>> # Create a GlobalExplainedClustering instance with global feature relevance
        >>> global_relevance = pd.Series([0.3, 0.5, 0.2], index=["A", "B", "C"])
        >>> global_explanation = GlobalExplainedClustering(global_relevance)
        >>> # Visualize global feature relevance
        >>> global_explanation.show_global_relevance()
        """
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


@dataclass()
class ClusterExplainedClustering:
    """
    A class for representing cluster-specific clustering explanations.

    Attributes:
        cluster_relevance (pd.DataFrame): A Pandas DataFrame containing cluster feature relevance scores.

    Methods:
        - __eq__(self, other) -> bool:
            Checks if two instances of ClusterExplainedClustering are equal.

        - show_cluster_relevance(self, subset_index):
            Visualizes cluster-wise feature relevance using a heatmap.

        - show_single_feature_relevance(self, feature, subset_index):
            Visualizes feature importance scores for a single feature across clusters using a bar plot.

        - show_single_cluster_relevance(self, cluster_index):
            Visualizes feature importance scores for a single cluster using a bar plot.

    Example:

    >>> # Create a ClusterExplainedClustering instance with cluster feature relevance
    >>> cluster_relevance_data = pd.DataFrame({
    ...     'feature_A': [0.3, 0.5, 0.2],
    ...     'feature_B': [0.2, 0.4, 0.6],
    ...     'feature_C': [0.4, 0.2, 0.5],
    ...     'assigned_clusters': [0, 1, 2]
    ... })
    >>> cluster_relevance_data.set_index('assigned_clusters', inplace=True)
    >>> cluster_explanation = ClusterExplainedClustering(cluster_relevance_data)
    >>> # Check if two instances of ClusterExplainedClustering are equal
    >>> another_cluster_relevance_data = pd.DataFrame({
    ...     'feature_A': [0.4, 0.4, 0.2],
    ...     'feature_B': [0.3, 0.3, 0.5],
    ...     'feature_C': [0.2, 0.1, 0.6],
    ...     'assigned_clusters': [0, 1, 2]
    ... })
    >>> another_cluster_relevance_data.set_index('assigned_clusters', inplace=True)
    >>> another_cluster_explanation = ClusterExplainedClustering(another_cluster_relevance_data)
    >>> cluster_explanation == another_cluster_explanation
    ... False
    >>> # Visualize cluster-wise feature relevance
    >>> cluster_explanation.show_cluster_relevance()
    >>> # Visualize feature importance scores for a single feature across clusters
    >>> cluster_explanation.show_single_feature_relevance("feature_A")
    >>> # Visualize feature importance scores for a single cluster
    >>> cluster_explanation.show_single_cluster_relevance(2)
    """

    cluster_relevance: pd.DataFrame

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClusterExplainedClustering):
            return False
        cluster_equal = np.all(
            np.isclose(self.cluster_relevance.values, other.cluster_relevance.values)
        )
        return cluster_equal  # type: ignore

    def show_cluster_relevance(self, subset_index: index_type = None):
        """
        Visualizes cluster-wise feature relevance using a heatmap.

        Args:
            subset_index: Optional list of cluster indices to subset the data.

        Example:

        >>> # Visualize cluster-wise feature relevance
        >>> cluster_explanation.show_cluster_relevance([0, 1])
        """
        relevances_to_plot = (
            self.cluster_relevance.T[subset_index].T
            if subset_index
            else self.cluster_relevance
        )
        sns.heatmap(relevances_to_plot, center=0, cmap="RdYlGn")
        plt.title("Clusterwise feature importance scores")
        plt.xlabel("Feature")
        plt.ylabel("Cluster")
        plt.show()

    def show_single_feature_relevance(
        self, feature: str, subset_index: index_type = None
    ):
        """
        Visualizes feature importance scores for a single feature across clusters using a bar plot.

        Args:
            feature: The name of a feature.
            subset_index: Optional list of cluster indices to subset the data.

        Example:

        >>> # Visualize feature importance scores for a single feature across clusters
        >>> cluster_explanation.show_single_feature_relevance("feature_A", [0, 1, 2])
        """
        feature_importance = (
            self.cluster_relevance.loc[subset_index, [feature]]  # type: ignore
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
        """
        Visualizes feature importance scores for a single cluster using a bar plot.

        Args:
            cluster_index: The index of the cluster.

        Example:

        >>> # Visualize feature importance scores for a single cluster
        >>> cluster_explanation.show_single_cluster_relevance(2)
        """
        observation_importance = self.cluster_relevance.loc[
            [cluster_index], :
        ].reset_index(drop=True)
        sns.barplot(
            x=0,
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


@dataclass()
class PointwiseExplainedClustering:
    """
    A class for representing pointwise clustering explanations.

    Attributes:
        pointwise_relevance (pd.DataFrame): A Pandas DataFrame containing pointwise feature relevance scores.

    Methods:
        - __eq__(self, other) -> bool:
            Checks if two instances of PointwiseExplainedClustering are equal.

        - show_pointwise_relevance(self, subset_index):
            Visualizes pointwise feature relevance using a heatmap.

        - show_single_feature_relevance(self, feature: str, subset_index):
            Visualizes feature importance scores for a single feature across observations using a bar plot.

        - show_single_observation_relevance(self, observation_index):
            Visualizes feature importance scores for a single observation using a bar plot.

    Example:

    >>> # Create a PointwiseExplainedClustering instance with pointwise feature relevance
    >>> pointwise_relevance_data = pd.DataFrame({
    ...     'feature_A': [0.3, 0.5, 0.2],
    ...     'feature_B': [0.2, 0.4, 0.6],
    ...     'feature_C': [0.4, 0.2, 0.5]
    ... })
    >>> pointwise_explanation = PointwiseExplainedClustering(pointwise_relevance_data)
    >>> # Check if two instances of PointwiseExplainedClustering are equal
    >>> another_pointwise_relevance_data = pd.DataFrame({
    ...     'feature_A': [0.4, 0.4, 0.2],
    ...     'feature_B': [0.3, 0.3, 0.5],
    ...     'feature_C': [0.2, 0.1, 0.6]
    ... })
    >>> another_pointwise_explanation = PointwiseExplainedClustering(another_pointwise_relevance_data)
    >>> pointwise_explanation == another_pointwise_explanation
    ... False
    >>> # Visualize pointwise feature relevance
    >>> pointwise_explanation.show_pointwise_relevance()
    >>> # Visualize feature importance scores for a single feature across observations
    >>> pointwise_explanation.show_single_feature_relevance("feature_A")
    >>> # Visualize feature importance scores for a single observation
    >>> pointwise_explanation.show_single_observation_relevance(2)
    """

    pointwise_relevance: pd.DataFrame

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PointwiseExplainedClustering):
            return False
        pointwise_equal = np.all(
            np.isclose(
                self.pointwise_relevance.values, other.pointwise_relevance.values
            )
        )
        return pointwise_equal  # type: ignore

    def show_pointwise_relevance(self, subset_index: index_type = None):
        """
        Visualizes pointwise feature relevance using a heatmap.

        Args:
            subset_index: Optional list of observation indices to subset the data.

        Example:

        >>> # Visualize pointwise feature relevance
        >>> pointwise_explanation.show_pointwise_relevance([0, 1, 2])
        """
        relevances_to_plot = (
            self.pointwise_relevance.loc[subset_index, :]  # type: ignore
            if subset_index
            else self.pointwise_relevance
        )
        sns.heatmap(relevances_to_plot, center=0, cmap="RdYlGn")
        plt.title("Point-wise feature importance scores")
        plt.xlabel("Feature")
        plt.ylabel("Observation")
        plt.show()

    def show_single_feature_relevance(
        self, feature: str, subset_index: index_type = None
    ):
        """
        Visualizes feature importance scores for a single feature across observations using a bar plot.

        Args:
            feature: The name of the feature.
            subset_index: Optional list of observation indices to subset the data.

        Example:

        >>> # Visualize feature importance scores for a single feature across observations
        >>> pointwise_explanation.show_single_feature_relevance("FeatureA", [0, 1, 2])
        """
        feature_importance = (
            self.pointwise_relevance.loc[subset_index, [feature]]  # type: ignore
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
        """
        Visualizes feature importance scores for a single observation using a bar plot.

        This method generates a bar plot to visualize feature importance scores for a single observation.

        Args:
            observation_index: The index of the observation.

        Example:

        >>> # Visualize feature importance scores for a single observation
        >>> pointwise_explanation.show_single_observation_relevance(2)
        """
        observation_importance = self.pointwise_relevance.loc[observation_index, :]
        sns.barplot(
            x=observation_index,
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
    """
    This class is used to represent clustering explanations, including global, pointwise, and cluster feature relevance.

    Methods:
        - __eq__(self, other) -> bool:
            Checks if two instances of ExplainedClustering are equal.

        - pointwise_relevance(self) -> Optional[PointwiseExplainedClustering]:
            Returns the pointwise feature relevance, if available.

        - cluster_relevance(self) -> Optional[ClusterExplainedClustering]:
            Returns the cluster feature relevance, if available.

        - global_relevance(self) -> GlobalExplainedClustering:
            Returns the global feature relevance.

        - pointwise_relevance_df(self) -> Optional[pd.DataFrame]:
            Returns the pointwise feature relevance as a DataFrame, if available.

        - cluster_relevance_df(self) -> Optional[pd.DataFrame]:
            Returns the cluster feature relevance as a DataFrame, if available.

        - global_relevance_df(self) -> pd.Series:
            Returns the global feature relevance as a Series.

        - show_pointwise_relevance(self, subset_index):
            Visualizes pointwise feature relevance using a heatmap.

        - show_pointwise_relevance_for_feature(self, feature: str, subset_index):
            Visualizes feature importance scores for a single feature across observations using a bar plot.

        - show_pointwise_relevance_for_observation(self, observation_index):
            Visualizes feature importance scores for a single observation using a bar plot.

        - show_cluster_relevance(self, subset_index):
            Visualizes cluster-wise feature relevance using a heatmap.

        - show_cluster_relevance_for_feature(self, feature, subset_index):
            Visualizes feature importance scores for a single feature across clusters using a bar plot.

        - show_cluster_relevance_for_cluster(self, cluster_index):
            Visualizes feature importance scores for a single cluster using a bar plot.

        - show_global_relevance(self):
            Visualizes global feature relevance using a bar plot.

    Example:

    >>> # Create an ExplainedClustering instance with global and pointwise feature relevance
    >>> global_relevance = pd.Series([0.3, 0.5, 0.2], index=["feature_A", "feature_B", "feature_C"])
    >>> pointwise_relevance_data = pd.DataFrame({
    ...     'feature_A': [0.3, 0.5, 0.2],
    ...     'feature_B': [0.2, 0.4, 0.6],
    ...     'feature_C': [0.4, 0.2, 0.5]
    ... })
    >>> explained_clustering = ExplainedClustering(global_relevance, pointwise_relevance_data)
    >>> # Check if two instances of ExplainedClustering are equal
    >>> another_global_relevance = pd.Series([0.4, 0.5, 0.2], index=["feature_A", "feature_B", "feature_C"])
    >>> another_pointwise_relevance_data = pd.DataFrame({
    ...     'feature_A': [0.4, 0.4, 0.2],
    ...     'feature_B': [0.3, 0.3, 0.5],
    ...     'feature_C': [0.2, 0.1, 0.6]
    ... })
    >>> another_explained_clustering = ExplainedClustering(another_global_relevance, another_pointwise_relevance_data)
    >>> explained_clustering == another_explained_clustering
    ... False
    >>> # Visualize pointwise feature relevance
    >>> explained_clustering.show_pointwise_relevance()
    >>> # Visualize feature importance scores for a single feature across observations
    >>> explained_clustering.show_pointwise_relevance_for_feature("feature_A")
    >>> # Visualize feature importance scores for a single observation
    >>> explained_clustering.show_pointwise_relevance_for_observation(2)
    """

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
        )  # type: ignore
        self._cluster_relevance = (
            ClusterExplainedClustering(cluster_relevance)
            if isinstance(cluster_relevance, pd.DataFrame)
            else None
        )  # type: ignore
        self._global_relevance = GlobalExplainedClustering(global_relevance)  # type: ignore

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
    def pointwise_relevance(self) -> Optional[PointwiseExplainedClustering]:
        """Returns PointwiseExplainedClustering if it exists."""
        self._check_relevance_exists(self._pointwise_relevance)
        return self._pointwise_relevance

    @property
    def cluster_relevance(self) -> Optional[ClusterExplainedClustering]:
        """Returns ClusterExplainedClustering if it exists."""
        self._check_relevance_exists(self._cluster_relevance)
        return self._cluster_relevance

    @property
    def global_relevance(self) -> GlobalExplainedClustering:
        """Returns GlobalExplainedClustering."""
        return self._global_relevance

    @property
    def pointwise_relevance_df(self) -> Optional[pd.DataFrame]:
        """Returns a dataframe containing the pointwise feature importances if they exist."""
        self._check_relevance_exists(self._pointwise_relevance)
        return self._pointwise_relevance.pointwise_relevance  # type: ignore

    @property
    def cluster_relevance_df(self) -> Optional[pd.DataFrame]:
        """Returns a dataframe containing the cluster-wise feature importances if they exist."""
        self._check_relevance_exists(self._cluster_relevance)
        return self._cluster_relevance.cluster_relevance  # type: ignore

    @property
    def global_relevance_df(self) -> pd.Series:
        """Returns a dataframe containing the global feature importances."""
        return self._global_relevance.global_relevance

    @staticmethod
    def _check_relevance_exists(
        explained_clustering: Optional[
            Union[PointwiseExplainedClustering, ClusterExplainedClustering]
        ] = None
    ):
        """
        Check whether the provided clustering expalantion exists

        Args:
            explained_clustering:
                Explained clustering object, which should be checked to exist.

        Raises:
            NonExistingRelevanceError: Raised if provided explained clustering does not exist.
        """
        if not (
            isinstance(explained_clustering, PointwiseExplainedClustering)
            or isinstance(explained_clustering, ClusterExplainedClustering)
        ):
            raise NonExistingRelevanceError(
                "The specific relevance score doesn't exist for your explainer!"
            )

    def show_pointwise_relevance(self, subset_index: index_type = None):
        """
        Visualizes pointwise feature relevance using a heatmap.

        Args:
            subset_index: Optional list of observation indices to subset the data.

        Example:

        >>> # Visualize pointwise feature relevance
        >>> explained_clustering.show_pointwise_relevance([0, 1, 2])
        """
        self.pointwise_relevance.show_pointwise_relevance(subset_index)  # type: ignore

    def show_pointwise_relevance_for_feature(
        self, feature: str, subset_index: index_type = None
    ):
        """
        Visualizes feature importance scores for a single feature across observations using a bar plot.

        Args:
            feature: The name of the feature.
            subset_index: Optional list of observation indices to subset the data.

        Example:

        >>> # Visualize feature importance scores for a single feature across observations
        >>> explained_clustering.show_pointwise_relevance_for_feature("FeatureA", [0, 1, 2])
        """
        self.pointwise_relevance.show_single_feature_relevance(feature, subset_index)  # type: ignore

    def show_pointwise_relevance_for_observation(self, observation_index: int):
        """
        Visualizes feature importance scores for a single observation using a bar plot.

        Args:
            observation_index: The index of the observation.

        Example:

        >>> # Visualize feature importance scores for a single observation
        >>> explained_clustering.show_pointwise_relevance_for_observation(2)
        """
        self.pointwise_relevance.show_single_observation_relevance(observation_index)  # type: ignore

    def show_cluster_relevance(self, subset_index: index_type = None):
        """
        Visualizes cluster-wise feature relevance using a heatmap.

        Args:
            subset_index: Optional list of cluster indices to subset the data.

        Example:

        >>> # Visualize cluster-wise feature relevance
        >>> explained_clustering.show_cluster_relevance([0, 1, 2])
        """
        self.cluster_relevance.show_cluster_relevance(subset_index)  # type: ignore

    def show_cluster_relevance_for_feature(
        self, feature: str, subset_index: index_type = None
    ):
        """
        Visualizes feature importance scores for a single feature across clusters using a bar plot.

        Args:
            feature: The name of the feature.
            subset_index: Optional list of cluster indices to subset the data.

        Example:

        >>> # Visualize feature importance scores for a single feature across clusters
        >>> explained_clustering.show_cluster_relevance_for_feature("FeatureA", [0, 1, 2])
        """
        self.cluster_relevance.show_single_feature_relevance(feature, subset_index)  # type: ignore

    def show_cluster_relevance_for_cluster(self, cluster_index: int):
        """
        Visualizes feature importance scores for a single cluster using a bar plot.

        Args:
            cluster_index: The index of the cluster.

        Example:

        >>> # Visualize feature importance scores for a single cluster
        >>> explained_clustering.show_cluster_relevance_for_cluster(2)
        """
        self.cluster_relevance.show_single_cluster_relevance(cluster_index)  # type: ignore

    def show_global_relevance(self):
        """
        Visualizes global feature relevance using a bar plot.

        Example:

        >>> # Visualize global feature relevance
        >>> explained_clustering.show_global_relevance()
        """
        self.global_relevance.show_global_relevance()


class BaseExplainer(ABC):
    """
    This is the base class for all cluster explainers, providing a common interface and functionality
    for clustering explanation.

    Methods:
        - fit(self): Abstract method for fitting the explainer. Subclasses must implement this method.
        - explain(self): Abstract method for generating cluster explanations. Subclasses must implement this method.
        - fit_explain(self): Convenience method that fits the explainer and immediately generates explanations.

    Attributes:
        - is_fitted (bool): Indicates whether the explainer has been fitted.
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
        """
        This method renames the feature columns in a DataFrame, providing more informative names
        when feature names are provided.

        If no feature names are provided every column is renamed to 'R<column number>'.

        Args:
            df: The DataFrame to rename columns.
            num_features: The number of feature columns.
            feature_names: A list of feature names (if provided).

        Returns:
            pd.DataFrame: The DataFrame with renamed columns.

        Raises:
            InconsistentNamingError: If the number of provided feature names does not match the number of features.

        Example:

        >>> df = pd.DataFrame({0: [1, 2, 3], 1: [4, 5, 6]})
        >>> num_features = 2
        >>> feature_names = ["feature_A", "feature_B"]
        >>> renamed_df = BaseExplainer._rename_feature_columns(df, num_features, feature_names)
        >>> renamed_df
        ...      feature_A  feature_B
        ... 0        1         4
        ... 1        2         5
        ... 2        3         6
        """
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
