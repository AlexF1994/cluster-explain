from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from nptyping import NDArray, Shape
from nptyping.typing_ import Floating, Int

from cxplain.base_explainer import BaseExplainer, ExplainedClustering
from cxplain.errors import NonExsitingXkmFlavourError
from cxplain.metrics import get_distance_metric

# TODO: - Write docstrings
#       - Write Docs
#       - CICD pipeline Github Actions
#       - sphinx Doku einfügen
#       - DataFrame als Input zulassen bei allen Explainern


class XkmExplainer(BaseExplainer):
    """
    Explain clustering results using the eXplainable K-Medoids (XKM) method, as proposed by (TBD).

    Attributes:
        data (NDArray[Shape["* num_obs, * num_features"], Floating]): Input data for clustering.
        cluster_centers (NDArray[Shape["* num_clusters, * num_features"], Floating]): Cluster centers.
        flavour (str): Flavour of the XkMedoids method.
        distance_metric (str): Distance metric used for calculating feature-wise distances.
        cluster_predictions (NDArray[Shape["* num_obs"], Int]): Cluster predictions for the input data.
        feature_names (Optional[List[str]]): Optional list of feature names.
        num_features (int): The number of features in the input data.
        feature_wise_distance_matrix (NDArray[Shape["* num_obs, * num_clusters,* num_features"], Floating]):
            Feature-wise distance matrix.

    Methods:
        - fit(self):
            Fits the explainer by calculating the feature-wise distance matrix making it ready for use.

        - _calculate_feature_wise_distance_matrix(self)
            -> NDArray[Shape["* num_obs, * num_clusters, * num_features"], Floating]:
            Calculates the feature-wise distance matrix.

        - _calculate_pointwise_relevance(self) -> pd.DataFrame:
            Computes pointwise feature relevance scores.

        - _calculate_cluster_relevance(self, pointwise_scores) -> pd.DataFrame:
            Computes cluster-wise feature relevance scores based on pointwise scores.

        - _calculate_global_relevance(self, pointwise_scores) -> pd.Series:
            Computes global feature relevance scores based on pointwise scores.

        - explain(self) -> ExplainedClustering:
            Explains clustering results by computing pointwise, cluster, and global feature relevance scores.

    Example:

    >>> # Create an XkmExplainer instance
    >>> data = ...  # Input data for clustering
    >>> cluster_centers = ...  # Cluster centers
    >>> flavour = ...  # Flavour of the XkMedoids method
    >>> distance_metric = ...  # Distance metric for calculating feature-wise distances
    >>> cluster_predictions = ...  # Cluster predictions for the input data
    >>> feature_names = ...  # Optional list of feature names
    >>> explainer = XkmExplainer(data, cluster_centers, flavour, distance_metric,
    ...                          cluster_predictions, feature_names=feature_names)
    >>> # Fit the explainer
    >>> explainer.fit()
    >>> # Explain clustering results
    >>> explained_result = explainer.explain()
    """

    def __init__(
        self,
        data: NDArray[Shape["* num_obs, * num_features"], Floating],  # type: ignore
        cluster_centers: NDArray[Shape["* num_clusters, * num_features"], Floating],  # type: ignore
        flavour: str,
        distance_metric: str,
        cluster_predictions: NDArray[Shape["* num_obs"], Int],  # type: ignore
        feature_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.distance_metric = distance_metric
        self.cluster_centers = cluster_centers
        self.flavour = _get_xkm_flavour(flavour)
        self.data = data
        self.cluster_predictions = cluster_predictions
        self.feature_wise_distance_matrix = None
        self.num_features = self.data.shape[1]
        self.feature_names = feature_names

    def _calculate_feature_wise_distance_matrix(
        self,
    ) -> NDArray[Shape["* num_obs, * num_clusters, * num_features"], Floating]:  # type: ignore
        """
        Calculates the feature-wise distance matrix of every feature of every observation to
        the corresponding feature coordinate of every cluster.

        Returns:
            NDArray[Shape["* num_obs, * num_clusters, * num_features"], Floating]: A distance tensor of shape
                                                                                   num_observations x num_clusters x
                                                                                   num_features.

        Example:

        >>> # Calculate the feature-wise distance matrix
        >>> feature_wise_distance_matrix = explainer._calculate_feature_wise_distance_matrix()
        """

        centers = np.array(
            [self.cluster_centers for observation_coordinates in self.data]
        )

        # calculate the distance of every feature value of ever obs to every feature value in every cluster.

        feature_wise_distance_matrix = []
        distance_metric = get_distance_metric(self.distance_metric)

        feature_wise_distance_matrix = [
            distance_metric.calculate(cluster_centers, observation_ccordinates)
            for cluster_centers, observation_ccordinates in zip(centers, self.data)
        ]

        return np.array(feature_wise_distance_matrix)

    def _calculate_pointwise_relevance(self) -> pd.DataFrame:
        """
        Computes pointwise feature relevance scores based on the XKM method.

        Returns:
            pd.DataFrame: Pointwise feature relevance scores.

        Example:

        >>> # Compute pointwise feature relevance scores
        >>> pointwise_relevance = explainer._calculate_pointwise_relevance()
        """
        pointwise_scores = self.flavour._calculate_pointwise_relevance(
            self.feature_wise_distance_matrix, self.cluster_predictions  # type: ignore
        )
        return pointwise_scores.pipe(
            self._rename_feature_columns, self.num_features, self.feature_names
        )

    def _calculate_cluster_relevance(
        self, pointwise_scores: pd.DataFrame
    ) -> pd.DataFrame:
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
        return (
            pointwise_scores.assign(assigned_clusters=self.cluster_predictions)
            .groupby(["assigned_clusters"])
            .mean()
        )

    def _calculate_global_relevance(self, pointwise_scores: pd.DataFrame) -> pd.Series:
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
        return pointwise_scores.mean()

    def fit(self):
        """
        Fits the explainer by calculating the feature-wise distance matrix making it ready for use.

        Example:

        >>> # Fit the explainer
        >>> explainer.fit()
        """
        if not self.is_fitted:
            self.feature_wise_distance_matrix = (
                self._calculate_feature_wise_distance_matrix()
            )
            self.is_fitted = True
        return self

    def explain(self) -> ExplainedClustering:
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


def _get_xkm_flavour(flavour: str, **kwargs):
    """
    Factory method for getting each flavour of Xkm.

    Args:
        flavour: The desired flavour of Xkm, either 'next_best' or 'all'.

    Returns:
        BaseXkmFlavour: An instance of the specified Xkm flavour.

    Raises:
        NonExistingXkmFlavourError: If the specified flavour does not exist.

    Example:

    >>> # Get an instance of the "next_best" Xkm flavour
    >>> xkm_flavour = _get_xkm_flavour("next_best")
    >>> # Get an instance of the "all" Xkm flavour
    >>> xkm_flavour = _get_xkm_flavour("all")
    >>> # Attempt to get an instance of a non-existing Xkm flavour (Raises an error)
    >>> xkm_flavour = _get_xkm_flavour("invalid_flavour")
    """
    if flavour == "next_best":
        return XkmNextBestFlavour()
    if flavour == "all":
        return XkmAllFlavour()
    else:
        raise NonExsitingXkmFlavourError(f"The flovour {flavour} doesn't exist.")


class BaseXkmFlavour(ABC):
    """
    Base class for different Xkm Flavours.

    This is an abstract base class for different flavours of the eXplainable k-medoids (Xkm) method.
    Subclasses should implement the _calculate_pointwise_relevance method to calculate pointwise feature relevance.

    Methods:
        - _calculate_pointwise_relevance(cls) -> pd.DataFrame:
            Abstract method to calculate pointwise feature relevance based on the XKM flavour.
    """

    @abstractmethod
    def _calculate_pointwise_relevance(cls) -> pd.DataFrame:
        pass


class XkmNextBestFlavour(BaseXkmFlavour):
    """
    This class calculates pointwise feature relevance for k-medoids clustering by finding the "next best"
    alternative cluster for each feature and observation.

    Methods:
        - _best_calc(
            feature_wise_distance_matrix, cluster_predictions) -> Tuple[NDArray, NDArray]:
            Find the "next best" alternative clusters for each feature and observation.

        - _calculate_pointwise_relevance(feature_wise_distance_matrix, cluster_predictions) -> pd.DataFrame:
            Calculate pointwise feature relevance using the distance to the "next best" cluster
            for each feature and observation.

    Example:

    >>> # Create an instance of XkmNextBestFlavour
    >>> xkm_flavour = XkmNextBestFlavour()
    >>> # Calculate pointwise feature relevance using the "Next Best" method
    >>> relevance_matrix = xkm_flavour._calculate_pointwise_relevance(
    ...     feature_wise_distance_matrix, cluster_predictions
    ... )
    """

    @staticmethod
    def _best_calc(
        feature_wise_distance_matrix: NDArray[
            Shape["* num_obs, * num_clusters, * num_features"], Floating  # type: ignore
        ],
        cluster_predictions: NDArray[Shape["* num_obs"], Int],  # type: ignore
    ) -> Tuple[NDArray, NDArray]:  # TODO: needs refactoring
        """
        Find the "next best" alternative clusters for each feature and observation.

        Args:
            feature_wise_distance_matrix (NDArray[Shape["* num_obs, * num_clusters, * num_features"], Floating]):
                Feature-wise distance matrix of every feature to every cluster for each observation.
            cluster_predictions (NDArray[Shape["* num_obs"], Int]):
                Assigned clusters for each observation.

        Returns:
            Tuple[NDArray, NDArray]: A tuple of two NDArrays, where the first contains the feature-based distances
            to the assigned cluster and the second contains the feature-based distances
            to the "next best" alternative cluster.
        """
        distance_matrix = feature_wise_distance_matrix

        num_features = distance_matrix.shape[2]

        assinged_cluster_list = []  # index des assigned cluster
        fb_distance_to_assinged_cluster_list = []  # fb feature based

        best_alterantive_list = []  # index des next best cluster
        fb_distance_to_best_alternative_list = []

        # for every obs:
        for idx, obs_distance_matrix in enumerate(
            distance_matrix
        ):  # e num_clusters x num_features
            # index of assinged cluster
            assigned_cluster = cluster_predictions[idx]  # für nte obs
            # feature-wise distances of point to assigned cluster
            distances_to_assigned = obs_distance_matrix[assigned_cluster]

            assinged_cluster_list.append(assigned_cluster)
            fb_distance_to_assinged_cluster_list.append(distances_to_assigned)

            # find best alternative:

            temp_bad = []  # best alternative distance
            temp_idx = []

            # for every feature
            for i in range(num_features):
                # best alternative:
                best_alternative_distance = min(
                    obs_distance_matrix[:, i]
                )  # minimum distance to cluster
                max_distance = max(obs_distance_matrix[:, i])
                x = obs_distance_matrix[
                    :, i
                ].tolist()  # only to get index --> to which cluster does it belong
                idx_best_alternative = x.index(
                    best_alternative_distance
                )  # welches cluster

                # if the best alternative is the assigned cluster, we have to find the second best alternative
                if idx_best_alternative == assigned_cluster:
                    # ensure second closest cluster is chosen as new min
                    x[idx_best_alternative] = x[idx_best_alternative] + max_distance
                    best_alternative_distance = min(x)
                    idx_best_alternative = x.index(best_alternative_distance)

                temp_bad.append(best_alternative_distance)
                temp_idx.append(idx_best_alternative)

            best_alterantive_list.append(temp_idx)
            fb_distance_to_best_alternative_list.append(temp_bad)

        return np.array(fb_distance_to_assinged_cluster_list), np.array(
            fb_distance_to_best_alternative_list
        )

    def _calculate_pointwise_relevance(
        self,
        feature_wise_distance_matrix: NDArray[
            Shape["* num_obs, * num_clusters, * num_features"], Floating  # type: ignore
        ],
        cluster_predictions: NDArray[Shape["* num_obs"], Int],  # type: ignore
    ) -> pd.DataFrame:  # TODO: needs refactoring
        """
        Calculate pointwise feature relevance using the distance to the
        "next best" cluster for each feature and observation.

        Args:
            feature_wise_distance_matrix (NDArray[Shape["* num_obs, * num_clusters, * num_features"], Floating]):
                Feature-wise distance matrix of every feature to every cluster for each observation.
            cluster_predictions (NDArray[Shape["* num_obs"], Int]):
                Assigned clusters for each observation.

        Returns:
            pd.DataFrame: A DataFrame containing pointwise feature relevance scores.

        Example:

        >>> # Create an instance of XkmNextBestFlavour
        >>> xkm_flavour = XkmNextBestFlavour()
        >>> # Prepare feature-wise distance matrix and cluster predictions
        >>> feature_wise_distance_matrix = np.array(...)  # Replace with your data
        >>> cluster_predictions = np.array(...)  # Replace with your data
        >>> # Calculate pointwise feature relevance using the "Next Best" method
        >>> relevance_matrix = xkm_flavour._calculate_pointwise_relevance(
        ...     feature_wise_distance_matrix, cluster_predictions
        ... )
        """
        fb_ac, fb_ba = self._best_calc(
            feature_wise_distance_matrix, cluster_predictions
        )
        pointwise_scores = (fb_ba - fb_ac) / (fb_ba + fb_ac)  # type: ignore
        return pd.DataFrame(pointwise_scores)


class XkmAllFlavour(BaseXkmFlavour):
    """
    This class calculates pointwise feature relevances for k-medoids clustering by comparing
    the distances to the actual assigned cluster with the complete distances over all clusters.

    Methods:
        - _calculate_pointwise_relevance(feature_wise_distance_matrix, cluster_predictions) -> pd.DataFrame:
            Calculate pointwise feature relevance using the distance to all clusters
            for each feature and observation.

    Example:

    >>> # Create an instance of XkmAllFlavour
    >>> xkm_flavour = XkmAllFlavour()
    >>> # Calculate pointwise feature relevance using the "All Features" method
    >>> feature_wise_distance_matrix = np.array(...)  # Replace with your data
    >>> cluster_predictions = np.array(...)  # Replace with your data
    >>> relevance_matrix = xkm_flavour._calculate_pointwise_relevance(
    ...     feature_wise_distance_matrix, cluster_predictions
    ... )
    """

    def _calculate_pointwise_relevance(
        self,
        feature_wise_distance_matrix: NDArray[
            Shape["* num_obs, * num_clusters, * num_features"], Floating  # type: ignore
        ],
        cluster_predictions: NDArray[Shape["* num_obs"], Int],  # type: ignore
    ) -> pd.DataFrame:
        """
        Calculate pointwise feature relevances using the distance to all clusters
        for each feature and observation.

        Args:
            feature_wise_distance_matrix (NDArray[Shape["* num_obs, * num_clusters, * num_features"], Floating]):
                Feature-wise distance matrix of every feature to every cluster for each observation.
            cluster_predictions (NDArray[Shape["* num_obs"], Int]):
                Assigned clusters for each observation.

        Returns:
            pd.DataFrame: A DataFrame containing pointwise feature relevance scores.

        Example:

        >>> # Create an instance of XkmAllFlavour
        >>> xkm_flavour = XkmAllFlavour()
        >>> # Prepare feature-wise distance matrix and cluster predictions
        >>> feature_wise_distance_matrix = np.array(...)  # Replace with your data
        >>> cluster_predictions = np.array(...)  # Replace with your data
        >>> # Calculate pointwise feature relevance using the "All Features" method
        >>> relevance_matrix = xkm_flavour._calculate_pointwise_relevance(
        ...     feature_wise_distance_matrix, cluster_predictions
        ... )
        """
        # sum up distances over cluster
        complete_distances = np.sum(feature_wise_distance_matrix, axis=1)
        # get distance to actual assigned cluster for every observation and feature
        relevant_distances = [
            feature_wise_distance_matrix[i, cluster_predictions[i], :]
            for i in range(feature_wise_distance_matrix.shape[0])
        ]
        actual_distances = np.vstack(
            relevant_distances
        )  # TODO: make own utility function as also used in shap
        # calculate relevance
        n_clusters = feature_wise_distance_matrix.shape[1]
        pointwise_scores = (
            complete_distances - n_clusters * actual_distances
        ) / complete_distances
        return pd.DataFrame(pointwise_scores)
