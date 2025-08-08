from typing import List, Optional

import pandas as pd
from nptyping import NDArray, Shape
from nptyping.typing_ import Floating
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted

from cxplain.base_explainer import BaseExplainer, ExplainedClustering
from cxplain.exkmc_helpers import Tree


class ExKMCExplainer(BaseExplainer):
    """
    A class for explaining clustering results using ExKMC (Explainable K-Means Clustering).

    This class is designed to explain clustering results using the ExKMC approach suggested in
    https://arxiv.org/abs/2002.12538, using the implementation found in https://github.com/navefr/ExKMC.
    You can configure the method by handing over corresponding kwargs when initializing an instance of this explainer.


    Attributes:
        data (NDArray[Shape["* num_obs, * num_features"], Floating]): Input data for clustering.
        kmeans (KMeans): A fitted K-Means clustering model.
        feature_names (Optional[List[str]]): Optional list of feature names.
        tree (Tree): ExKMC explainer for feature importance computation.
        num_features (int): The number of features in the input data.

    Methods:
        - fit(self) -> ExKMCExplainer:
            Fits the ExKMC explainer to the input data and the fitted K-Means model, making it ready for use.

        - explain(self) -> ExplainedClustering:
            Computes and returns global feature relevance scores based on the fitted ExKMC explainer.

    Example:

    >>> # Create an ExKMCExplainer instance
    >>> data = ...  # Input data for clustering
    >>> kmeans_model = ...  # A fitted K-Means model
    >>> feature_names = ...  # Optional list of feature names
    >>> explainer = ExKMCExplainer(data, kmeans_fitted=kmeans_model, feature_names=feature_names)
    >>> # Fit the explainer
    >>> explainer.fit()
    >>> # Explain the clustering results
    >>> explained_result = explainer.explain()
    """

    def __init__(
        self,
        data: NDArray[Shape["* num_obs, * num_features"], Floating],  # type: ignore
        kmeans_fitted: KMeans,
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__()
        self.data = data
        self.kmeans = kmeans_fitted
        self.tree = Tree(**kwargs)
        self.num_features = self.data.shape[1]
        self.feature_names = feature_names

    def fit(self):
        """
        Fits the ExKMC explainer to the input data and the fitted K-Means model, making it ready for use.

        Returns:
            ExKMCExplainer: The fitted ExKMCExplainer instance.

        Example:

        >>> # Fit the explainer
        >>> explainer.fit()
        """
        check_is_fitted(self.kmeans)
        self.tree.fit(self.data, self.kmeans)
        self.is_fitted = True
        return self

    def _calculate_global_relevance(self) -> pd.DataFrame:
        """
        Calculates and returns global feature relevance scores based on the fitted ExKMC explainer.

        Returns:
            pd.DataFrame: A DataFrame containing global feature relevance scores.

        Example:

        >>> # Calculate global feature relevance scores
        >>> global_relevance = explainer._calculate_global_relevance()
        """
        feature_importances = self.tree._feature_importance
        return pd.DataFrame(feature_importances).T.pipe(
            self._rename_feature_columns, self.num_features, self.feature_names
        )

    def explain(self) -> ExplainedClustering:
        """
        Computes and returns global feature relevance scores based on the fitted ExKMC explainer.

        Returns:
            ExplainedClustering: An instance of ExplainedClustering containing global feature relevance scores.

        Raises:
            NotFittedError: If the explainer is not fitted (fit method not called).

        Example:

        >>> # Explain the clustering results
        >>> explained_result = explainer.explain()
        """
        self._check_fitted()
        feature_importances = self._calculate_global_relevance()
        return ExplainedClustering(
            global_relevance=feature_importances.squeeze()
        )  # pyright: ignore[reportArgumentType]
