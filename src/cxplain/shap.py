import numpy as np
import pandas as pd
from nptyping import NDArray, Shape
from nptyping.typing_ import Floating, Int
from shap import TreeExplainer
from sklearn.ensemble import RandomForestClassifier

from cxplain.base_explainer import BaseExplainer, ExplainedClustering


class ShapExplainer(BaseExplainer):
    """Right now only Random Forest is available as base classifier """
    
    def __init__(self, data: NDArray[Shape["* num_obs, * num_features"], Floating], 
                 cluster_predictions: NDArray[Shape["* num_obs"], Int],
                 feature_names: Optional[List[str]] = None,,
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
        self.forest.fit(self.data, self.cluster_predictions)
        self.is_fitted = True
        self.explainer = TreeExplainer(self.forest)
        return self
        
    def _calculate_pointwise_relevance(self) -> pd.DataFrame: 
        self._check_fitted()
        shap_values = np.array(self.explainer.shap_values(self.data))
        relevant_shap_values = self._get_relevant_shap_values(shap_values, self.cluster_predictions)
        return (pd.DataFrame(relevant_shap_values)
                .pipe(self._rename_feature_columns, self.num_features, self.feature_names))
        
    @staticmethod
    def _get_relevant_shap_values(shap_values:  NDArray[Shape["* num_cluster, * num_obs, * num_features"], Floating],
                                  cluster_predictions: NDArray[Shape["* num_obs"], Int]):
        """
        Extract only the shapley value for the assigned cluster, as relevance for existing clustering
        is of primary interest.
        """
        relevant_rows = [shap_values[cluster_predictions[i], i, :] for i in range(shap_values.shape[1])]
        return np.vstack(relevant_rows)

    def _calculate_cluster_relevance(self, pointwise_scores: pd.DataFrame) -> pd.DataFrame: # TODO: duplicated code
        self._check_fitted()
        return (pointwise_scores
                .pipe(self._rename_feature_columns, self.num_features)
                .assign(assigned_clusters=self.cluster_predictions)
                .groupby(["assigned_clusters"])
                .mean())

    def _calculate_global_relevance(self, pointwise_scores: pd.DataFrame) -> pd.Series: # TODO: duplicated code
        return pointwise_scores.mean()
    
    def explain(self) -> ExplainedClustering:
        self._check_fitted() 
        pointwise_relevance = self._calculate_pointwise_relevance()
        cluster_relevance = self._calculate_cluster_relevance(pointwise_scores=pointwise_relevance)
        global_relevance = self._calculate_global_relevance(pointwise_scores=pointwise_relevance)  

        return ExplainedClustering(pointwise_relevance=pointwise_relevance,
                                   cluster_relevance=cluster_relevance,
                                   global_relevance=global_relevance)

        