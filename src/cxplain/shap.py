from typing import List, Optional
import numpy as np
import pandas as pd
from nptyping import NDArray, Shape
from nptyping.typing_ import Floating, Int
from shap import TreeExplainer
from sklearn.ensemble import RandomForestClassifier

from cxplain.base_explainer import BaseExplainer, ExplainedClustering


class ShapExplainer(BaseExplainer):
    """
    Explain clustering using SHAP (SHapley Additive exPlanations) values,
    currently supporting Random Forest as the base classifier as well as the TreeExplainer for explanations.
    You can configure the base classifier by handing over corresponding kwargs
    when initializing an instance of this explainer.

    Attributes:
        data (NDArray[Shape["* num_obs, * num_features"], Floating]): Input data for clustering.
        cluster_predictions (NDArray[Shape["* num_obs"], Int]): Cluster predictions for the input data.
        feature_names (Optional[List[str]]): Optional list of feature names.
        num_features (int): The number of features in the input data.
        forest (RandomForestClassifier): Random Forest classifier used for SHAP value calculation.
        explainer (TreeExplainer): The SHAP explainer instance.

    Methods:
        - fit(self):
            Fits the explainer by training the Random Forest classifier and initializing the SHAP explainer.

        - _calculate_pointwise_relevance(self) -> pd.DataFrame:
            Computes pointwise feature relevance scores using SHAP values.

        - _get_relevant_shap_values(
            shap_values: NDArray[Shape["* num_cluster, * num_obs, * num_features"], Floating],
            cluster_predictions: NDArray[Shape["* num_obs"], Int]
        ) -> NDArray[Shape["* num_obs, * num_features"], Floating]:
            Extracts only the SHAP values of to the assigned cluster for each observation.

        - _calculate_cluster_relevance(self, pointwise_scores: pd.DataFrame) -> pd.DataFrame:
            Computes cluster-wise feature relevance scores based on pointwise scores.

        - _calculate_global_relevance(self, pointwise_scores: pd.DataFrame) -> pd.Series:
            Computes global feature relevance scores based on pointwise scores.

        - explain(self) -> ExplainedClustering:
            Explains clustering results by computing pointwise, cluster, and global feature relevance scores.

    Example:
    >>> # Create a ShapExplainer instance
    >>> data = ...  # Input data for clustering
    >>> cluster_predictions = ...  # Cluster predictions for the input data
    >>> feature_names = ...  # Optional list of feature names
    >>> explainer = ShapExplainer(data, cluster_predictions, feature_names=feature_names)

    >>> # Fit the explainer
    >>> explainer.fit()

    >>> # Explain clustering results
    >>> explained_result = explainer.explain()
    """

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
        self.explainer = None
        self.feature_names = feature_names

    def fit(self):
        """
        Fits the explainer by training the Random Forest classifier and initializing the SHAP explainer.

        Example:
        >>> # Fit the explainer
        >>> explainer.fit()
        """
        self.forest.fit(self.data, self.cluster_predictions)
        self.is_fitted = True
        self.explainer = TreeExplainer(self.forest)  # type: ignore
        return self

    def _calculate_pointwise_relevance(self) -> pd.DataFrame:
        """
        Computes pointwise feature relevance scores using SHAP values.

        Returns:
            pd.DataFrame: Pointwise feature relevance scores based on SHAP values.

        Example:
        >>> # Compute pointwise feature relevance scores
        >>> pointwise_relevance = explainer._calculate_pointwise_relevance()
        """
        self._check_fitted()
        shap_values = np.array(self.explainer.shap_values(self.data))
        relevant_shap_values = self._get_relevant_shap_values(
            shap_values, self.cluster_predictions
        )
        return pd.DataFrame(relevant_shap_values).pipe(
            self._rename_feature_columns, self.num_features, self.feature_names
        )

    @staticmethod
    def _get_relevant_shap_values(
        shap_values: NDArray[
            Shape["* num_cluster, * num_obs, * num_features"], Floating  # type: ignore
        ],
        cluster_predictions: NDArray[Shape["* num_obs"], Int],  # type: ignore
    ):
        """
        Extracts only the SHAP values of to the assigned cluster for each observation.

        Args:
            shap_values (NDArray[Shape["* num_cluster, * num_obs, * num_features"], Floating]): SHAP values.
            cluster_predictions (NDArray[Shape["* num_obs"], Int]): Cluster predictions for the input data.

        Returns:
            NDArray[Shape["* num_obs, * num_features"], Floating]: Relevant SHAP values.

        Example:
        >>> # Extract relevant SHAP values
        >>> relevant_shap_values = explainer._get_relevant_shap_values(shap_values, cluster_predictions)
        """
        relevant_rows = [
            shap_values[cluster_predictions[i], i, :]
            for i in range(shap_values.shape[1])
        ]
        return np.vstack(relevant_rows)

    def _calculate_cluster_relevance(
        self, pointwise_scores: pd.DataFrame
    ) -> pd.DataFrame:  # TODO: duplicated code
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
            pointwise_scores.assign(assigned_clusters=self.cluster_predictions)
            .groupby(["assigned_clusters"])
            .mean()
        )

    def _calculate_global_relevance(
        self, pointwise_scores: pd.DataFrame
    ) -> pd.Series:  # TODO: duplicated code
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
        return pointwise_scores.mean()

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
