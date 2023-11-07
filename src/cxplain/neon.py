from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import mean
from typing import List, Optional

import numpy as np
import pandas as pd
from nptyping import Floating, Int, NDArray, Shape

from cxplain.base_explainer import BaseExplainer, ExplainedClustering
from cxplain.errors import NotFittedError


class NeonExplainer(BaseExplainer, ABC):
    """
    A base class for explaining clustering results using the Neon (neuralization-propagation) method
    suggested in https://arxiv.org/abs/1906.07633. Subclasses are required to implement
    the abstract method _init_network for custom network initialization.

    Attributes:
        data (NDArray[Shape["* num_obs, * num_features"], Floating]): Input data for clustering.
        cluster_centers (NDArray[Shape["* num_clusters,
                         * num_features"], Floating]): Cluster centers for the input data.
        predictions (NDArray[Shape["* num_obs"], Int]): Cluster predictions for the input data.
        feature_names (Optional[List[str]]): Optional list of feature names.
        num_clusters (int): The number of clusters.
        num_features (int): The number of features in the input data.
        networks (List[NeonNetwork]): List of NeonNetwork instances.

    Methods:
        - _init_network(self):
            Abstract method to initialize the neural network for explaining clustering results.
            Subclasses must implement this method.
    """

    def __init__(
        self,
        data: NDArray[Shape["* num_obs, * num_features"], Floating],  # type: ignore
        cluster_centers: NDArray[Shape["* num_clusters, * num_features"], Floating],  # type: ignore
        predictions: NDArray[Shape["* num_obs"], Int],  # type: ignore
        feature_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.cluster_centers = cluster_centers
        self.num_clusters = self.cluster_centers.shape[0]
        self.data = data
        self.predictions = predictions
        self.feature_names = feature_names
        self.num_features = self.data.shape[1]
        self.networks = []

    @abstractmethod
    def _init_network(self):
        pass


@dataclass
class KMeansNetwork:
    """
    A class representing a neuralized K-Means clustering, as described in https://arxiv.org/abs/1906.07633.

    Attributes:
        index_actual (int): The index of the actual cluster.
        weights (NDArray[Shape["* num_clusters, * num_features"], Floating]): Weights of the neural network.
        bias (NDArray[Shape["* num_clusters"], Floating]): Bias terms of the neural network.
        hidden_layer (Optional[NDArray[Shape["* num_clusters"], Floating]]): The hidden layer of the neural network.
        output (Optional[float]): The output of the neural network.

    Methods:
        - forward(self, observation: NDArray[Shape["* num_features"], Floating]) -> KMeansNetwork:
            Performs a forward pass of the neural network with the given observation and computes the output.

        - backward(self, observation: NDArray[Shape["* num_features"], Floating], beta: float)
          -> NDArray[Shape["* num_features"], Floating]:
            Performs a backward pass of the neural network and computes feature
            relevance scores using LRP (Layer-wise Relevance Propagation).

        - _check_forward_pass(self):
            Checks if a forward pass has been conducted, raising an error if not.

    Example:
    >>> # Create a KMeansNetwork instance
    >>> index_actual = 0
    >>> weights = np.random.rand(3, 2)  # Weights for the neural network
    >>> bias = np.random.rand(3)  # Bias terms for the neural network
    >>> network = KMeansNetwork(index_actual, weights, bias)

    >>> # Perform a forward pass
    >>> observation = np.random.rand(2)  # Input observation
    >>> network.forward(observation)

    >>> # Perform a backward pass to compute feature relevances
    >>> beta = 0.5  # Beta value for the backward pass
    >>> feature_relevances = network.backward(observation, beta)

    >>> # Check if a forward pass has been conducted
    >>> network._check_forward_pass()
    """

    index_actual: int
    weights: NDArray[Shape["* num_clusters, * num_features"], Floating]  # type: ignore
    bias: NDArray[Shape["* num_clusters"], Floating]  # type: ignore
    hidden_layer: Optional[NDArray[Shape["* num_clusters"], Floating]] = None  # type: ignore
    output: Optional[float] = None

    def forward(
        self, observation: NDArray[Shape["* num_features"], Floating]  # type: ignore
    ) -> "KMeansNetwork":
        """
        Performs a forward pass of the neuralized K-Means network with the given observation and computes the output.

        Args:
            observation (NDArray[Shape["* num_features"], Floating]): Input observation for the forward pass.

        Returns:
            KMeansNetwork: The KMeansNetwork instance after the forward pass.

        Example:
        >>> # Perform a forward pass
        >>> observation = np.random.rand(2)  # Input observation
        >>> network.forward(observation)
        """
        self.hidden_layer = self.weights.dot(observation) + self.bias
        self.output = np.amin(np.delete(self.hidden_layer, self.index_actual))
        return self

    def backward(
        self, observation: NDArray[Shape["* num_features"], Floating], beta: float  # type: ignore
    ) -> NDArray[Shape["* num_features"], Floating]:  # type: ignore
        """
        Performs a backward pass of the neuralized K-Means network and computes feature relevance scores for one
        observation using LRP (Layer-wise Relevance Propagation).

        Args:
            observation (NDArray[Shape["* num_features"], Floating]): Input observation for the backward pass.
            beta (float): Beta value for the backward pass.

        Returns:
            NDArray[Shape["* num_features"], Floating]: Feature relevance scores computed during the backward pass.

        Example:
        >>> # Perform a backward pass to compute feature relevances
        >>> beta = 0.5  # Beta value for the backward pass
        >>> feature_relevances = network.backward(observation, beta)
        """
        self._check_forward_pass()

        num_clusters = self.hidden_layer.shape[0]
        num_features = observation.shape[0]

        # relevance intermediate layer
        relevance_intermediate = (
            np.exp(-beta * self.hidden_layer)  # type: ignore
            / (np.sum(np.exp(-beta * np.delete(self.hidden_layer, self.index_actual))))  # type: ignore
        ) * self.output
        # relevance input
        centers_distance = self.weights / 4
        centers_distance_wo_actual = np.delete(
            centers_distance, self.index_actual, axis=0
        )
        hidden_wo_actual = np.delete(self.hidden_layer, self.index_actual, axis=0)  # type: ignore

        contribution = np.multiply(
            (
                np.vstack([observation] * (num_clusters - 1))
                - centers_distance_wo_actual
            ),
            np.vstack([hidden_wo_actual] * num_features).T,
        )
        sum_contribution = np.sum(contribution, axis=0)

        relevance_intermediate_wo_actual = np.delete(
            relevance_intermediate, self.index_actual, axis=0
        )
        cluster_contribution = np.multiply(
            np.vstack([relevance_intermediate_wo_actual] * num_features).T,
            (contribution / sum_contribution),
        )
        feature_relevances = np.sum(cluster_contribution, axis=0)

        return feature_relevances

    def _check_forward_pass(self):
        """
        Checks if a forward pass has been conducted, raising an error if not.

        Raises:
            NotFittedError: Raised if no forward pass has been conducted beforehand.

        Notes:
            This method is intended for internal use and should not be called directly.

        Example:
        >>> # Check if a forward pass has been conducted
        >>> network._check_forward_pass()
        """
        if not (isinstance(self.hidden_layer, NDArray) or isinstance(self.output, int)):
            raise NotFittedError("You have to conduct a forward pass first!")


class NeonKMeansExplainer(
    NeonExplainer
):  # TODO: Maybe use factory method for different Neon Exlpainers
    """
    This class uses the Neon (neuralization-propagation) method, suggested
    in https://arxiv.org/abs/1906.07633, to explain K-Means clustering results by computing pointwise,
    cluster, and global feature relevance scores.

    Attributes:
        data (NDArray[Shape["* num_obs, * num_features"], Floating]): Input data for clustering.
        cluster_centers (NDArray[Shape["* num_clusters, * num_features"], Floating]): Cluster centers
                                                                                      for the input data.
        predictions (NDArray[Shape["* num_obs"], Int]): Cluster predictions for the input data.
        feature_names (Optional[List[str]]): Optional list of feature names.
        num_clusters (int): The number of clusters.
        num_features (int): The number of features in the input data.
        networks (List[KMeansNetwork]): List of KMeansNetwork instances for explaining clustering results.

    Methods:
        - _init_network(self, index_observation: int) -> KMeansNetwork:
            Initializes and returns a KMeansNetwork for a specific observation.

        - _calculate_pointwise_relevance(self) -> pd.DataFrame:
            Computes pointwise feature relevance scores for the input data.

        - _calculate_cluster_relevance(self, pointwise_scores: pd.DataFrame) -> pd.DataFrame:
            Computes cluster-wise feature relevance scores based on pointwise scores.

        - _calculate_global_relevance(self, pointwise_scores: pd.DataFrame) -> pd.Series:
            Computes global feature relevance scores based on pointwise scores.

        - _get_beta(self) -> float:
            Computes the beta value for relevance calculations.

        - fit(self):
            Fits the explainer by initializing KMeansNetwork instances for each observation.

        - explain(self) -> ExplainedClustering:
            Explains clustering results by computing pointwise, cluster, and global feature relevance scores.

    Example:
    >>> # Create a NeonKMeansExplainer instance
    >>> data = ...  # Input data for clustering
    >>> cluster_centers = ...  # Cluster centers for the input data
    >>> predictions = ...  # Cluster predictions for the input data
    >>> feature_names = ...  # Optional list of feature names
    >>> explainer = NeonKMeansExplainer(data, cluster_centers, predictions, feature_names=feature_names)

    >>> # Fit the explainer
    >>> explainer.fit()

    >>> # Explain clustering results
    >>> explained_result = explainer.explain()
    """

    def _init_network(self, index_observation: int) -> KMeansNetwork:
        """
        Initializes and returns a KMeansNetwork for a specific observation.

        Args:
            index_observation (int): Index of the observation for which to initialize the network.

        Returns:
            KMeansNetwork: Initialized KMeansNetwork instance.

        Example:
        >>> # Initialize a KMeansNetwork for a specific observation
        >>> network = explainer._init_network(index_observation)
        """
        index_actual = self.predictions[index_observation]
        center_actual = self.cluster_centers[index_actual]
        centers_actual = np.vstack([center_actual] * self.num_clusters)
        weights = 2 * (centers_actual - self.cluster_centers)
        bias = (
            np.linalg.norm(self.cluster_centers, ord=2, axis=1) ** 2
            - np.linalg.norm(centers_actual, ord=2, axis=1) ** 2
        )

        return KMeansNetwork(index_actual=index_actual, weights=weights, bias=bias)

    def _calculate_pointwise_relevance(self) -> pd.DataFrame:
        """
        Computes pointwise feature relevance scores for the input data.

        Returns:
            pd.DataFrame: Pointwise feature relevance scores.

        Example:
        >>> # Compute pointwise feature relevance scores
        >>> pointwise_relevance = explainer._calculate_pointwise_relevance()
        """
        self._check_fitted()  # TODO: fitted decorator
        beta = self._get_beta()
        relevances = [
            self.networks[index].backward(observation, beta)
            for index, observation in enumerate(self.data)
        ]
        return pd.DataFrame(np.row_stack(relevances)).pipe(
            self._rename_feature_columns, self.num_features, self.feature_names
        )

    def _calculate_cluster_relevance(
        self, pointwise_scores: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Computes cluster-wise feature relevance scores based on pointwise scores.

        Args:
            pointwise_scores (pd.DataFrame): Pointwise feature relevance scores.

        Returns:
            pd.DataFrame: Cluster-wise feature relevance scores.

        Example:
        >>> # Compute cluster-wise feature relevance scores
        >>> cluster_relevance = explainer._calculate_cluster_relevance(pointwise_scores)
        """
        self._check_fitted()
        return (
            pointwise_scores.assign(assigned_clusters=self.predictions)
            .groupby(["assigned_clusters"])
            .mean()
        )

    def _calculate_global_relevance(self, pointwise_scores: pd.DataFrame) -> pd.Series:
        """
        Computes global feature relevance scores based on pointwise scores.

        Args:
            pointwise_scores (pd.DataFrame): Pointwise feature relevance scores.

        Returns:
            pd.Series: Global feature relevance scores.

        Example:
        >>> # Compute global feature relevance scores
        >>> global_relevance = explainer._calculate_global_relevance(pointwise_scores)
        """
        self._check_fitted()
        return pointwise_scores.mean()

    def _get_beta(self) -> float:
        """
        Computes the beta value for relevance calculations as 1 divided by the average of the
        outputs of the neuralized K-Means network, as described in https://arxiv.org/abs/1906.07633.

        Returns:
            float: The beta value.

        Example:
        >>> # Compute the beta value
        >>> beta = explainer._get_beta()
        """
        return 1 / mean(
            [self.networks[index].output for index in range(self.data.shape[0])]
        )

    def fit(self):
        """
        Fits the explainer by initializing KMeansNetwork instances for each observation.

        Example:
        >>> # Fit the explainer
        >>> explainer.fit()
        """
        if not self.is_fitted:
            for index, observation in enumerate(self.data):
                self.networks.append(self._init_network(index).forward(observation))
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
