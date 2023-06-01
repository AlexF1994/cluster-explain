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
    def __init__(self, data: NDArray[Shape["* num_obs, * num_features"], Floating], 
                 cluster_centers: NDArray[Shape["* num_clusters, * num_features"], Floating],
                 predictions: NDArray[Shape["* num_obs"], Int],
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
    index_actual: int
    weights: NDArray[Shape["* num_clusters, * num_features"], Floating]
    bias: NDArray[Shape["* num_clusters"], Floating]
    hidden_layer: Optional[NDArray[Shape["* num_clusters"], Floating]] = None
    output: Optional[float] = None
                     
    def forward(self, observation: NDArray[Shape["* num_features"], Floating]
                ) -> "KMeansNetwork":
        self.hidden_layer = self.weights.dot(observation) + self.bias
        self.output = np.amin(np.delete(self.hidden_layer, self.index_actual))
        return self
    
    def backward(self, observation: NDArray[Shape["* num_features"], Floating], beta: float
                 ) -> NDArray[Shape["* num_features"], Floating]:
        self._check_forward_pass()
        
        num_clusters = self.hidden_layer.shape[0]
        num_features = observation.shape[0]
        
        # relevance intermediate layer
        relevance_intermediate = ((np.exp(-beta * self.hidden_layer) / 
                                   (np.sum(np.exp(-beta * np.delete(self.hidden_layer, self.index_actual)))))
                                  * self.output)
        # relevance input
        centers_distance = self.weights / 4
        centers_distance_wo_actual = np.delete(centers_distance, self.index_actual, axis=0)
        hidden_wo_actual = np.delete(self.hidden_layer, self.index_actual, axis=0)
        
        contribution = np.multiply((np.vstack([observation] * (num_clusters - 1)) - centers_distance_wo_actual), 
                                   np.vstack([hidden_wo_actual]* num_features).T)
        sum_contribution = np.sum(contribution, axis=0)
        
        relevance_intermediate_wo_actual = np.delete(relevance_intermediate, self.index_actual, axis=0)
        cluster_contribution = np.multiply(np.vstack([relevance_intermediate_wo_actual] * num_features).T, 
                                           (contribution / sum_contribution))
        feature_relevances = np.sum(cluster_contribution, axis=0)
        
        return feature_relevances
        
    def _check_forward_pass(self):
        if not (isinstance(self.hidden_layer, NDArray) or isinstance(self.output, int)):
            raise NotFittedError("You have to conduct a forward pass first!")
    
    
class NeonKMeansExplainer(NeonExplainer):  # TODO: Maybe use factory method for different Neon Exlpainers 
    
    def _init_network(self, index_observation: int) -> KMeansNetwork:
        index_actual = self.predictions[index_observation]
        center_actual = self.cluster_centers[index_actual]
        centers_actual = np.vstack([center_actual] * self.num_clusters)
        weights = 2 * (centers_actual - self.cluster_centers)
        bias = np.linalg.norm(self.cluster_centers, ord=2, axis=1)**2 - np.linalg.norm(centers_actual, ord=2, axis=1)**2
        
        return KMeansNetwork(index_actual=index_actual,
                             weights=weights,
                             bias=bias)
    
    def _calculate_pointwise_relevance(self) -> pd.DataFrame:
        self._check_fitted() # TODO: fitted decorator
        beta = self._get_beta()
        relevances = [self.networks[index].backward(observation, beta) 
                      for index , observation in enumerate(self.data)]
        return (pd.DataFrame(np.row_stack(relevances))
                .pipe(self._rename_feature_columns, self.num_features, self.feature_names))
        
    def _calculate_cluster_relevance(self, pointwise_scores: pd.DataFrame) -> pd.DataFrame: # TODO: average decorator
        self._check_fitted()
        return (pointwise_scores
                .assign(assigned_clusters=self.predictions)
                .groupby(["assigned_clusters"])
                .mean())
        
    def _calculate_global_relevance(self, pointwise_scores: pd.DataFrame) -> pd.Series:
        self._check_fitted()
        return pointwise_scores.mean()
        
    def _get_beta(self) -> float:
        return 1 / mean([self.networks[index].output for index in range(self.data.shape[0])])          
       
    def fit(self):
        if not self.is_fitted:
            for index, observation in enumerate(self.data):
                self.networks.append(self._init_network(index).forward(observation))
            self.is_fitted = True
        return self

    def explain(self) -> ExplainedClustering:
        self._check_fitted() 
        pointwise_relevance = self._calculate_pointwise_relevance()
        cluster_relevance = self._calculate_cluster_relevance(pointwise_scores=pointwise_relevance)  
        global_relevance = self._calculate_global_relevance(pointwise_scores=pointwise_relevance) 
        
        return ExplainedClustering(pointwise_relevance=pointwise_relevance,
                                   cluster_relevance=cluster_relevance,
                                   global_relevance=global_relevance)
