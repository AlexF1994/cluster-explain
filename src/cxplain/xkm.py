import numpy as np
import pandas as pd
from nptyping import NDArray, Shape
from nptyping.typing_ import Floating, Int

from cxplain.base_explainer import BaseExplainer, ExplainedClustering
from cxplain.errors import NotFittedError
from cxplain.metrics import get_distance_metric

# TODO: - Write docstrings
#       - Write Docs
#       - CICD pipeline Github Actions
#       - sphinx Doku einfügen
#       - fit hast to return self!!
    
class Xkm(BaseExplainer):

    "eXplainable k-medoids"
    
    def __init__(self, data: NDArray[Shape["* num_obs, * num_features"], Floating], 
                 cluster_centers: NDArray[Shape["* num_clusters, * num_features"], Floating],
                 distance_metric: str, 
                 cluster_predictions: NDArray[Shape["* num_obs"], Int]
                 ):
        super().__init__()
        self.distance_metric = distance_metric
        self.cluster_centers = cluster_centers
        self.data = data
        self.cluster_predictions = cluster_predictions
        self.feature_wise_distance_matrix = None
        self.num_features = self.data.shape[1]
        
    
    def _calculate_feature_wise_distance_matrix(self
        ) -> NDArray[Shape["* num_obs, * num_clusters, * num_features"], Floating]:
        """
        Calculate feature-wise distance matrix of every feature of every observation to
        the corresponding feature coordinate of every cluster.

        Returns:
            NDArray[Any, Floating]: A distance tensor of shape num_observations x num_clusters x num_features
        """
         
        centers = np.array([self.cluster_centers for observation_coordinates in self.data])

        # calculate the distance of every feature value of ever obs to every feature value in every cluster.
    
        feature_wise_distance_matrix = []
        distance_metric = get_distance_metric(self.distance_metric)

        feature_wise_distance_matrix = [distance_metric.calculate(cluster_centers, observation_ccordinates)
                                        for cluster_centers, observation_ccordinates 
                                        in zip(centers, self.data)]                
    
        return np.array(feature_wise_distance_matrix)
    
    def _best_calc(self) -> tuple[NDArray, NDArray]: # TODO: needs refactoring
        try :
            distance_matrix = self.feature_wise_distance_matrix
        except AttributeError as err:
            raise NotFittedError("You have to calculate th feature wise distance matrix first!") from err

        num_features = distance_matrix.shape[2]

        assinged_cluster_list = [] # index des assigned cluster 
        fb_distance_to_assinged_cluster_list = [] #fb feature based
    
        best_alterantive_list = [] # index des next best cluster 
        fb_distance_to_best_alternative_list = []
    
        #for every obs:
        for idx, obs_distance_matrix in enumerate(distance_matrix): # e num_clusters x num_features
            #index of assinged cluster
            assigned_cluster = self.cluster_predictions[idx] # für nte obs
            #feature-wise distances of point to assigned cluster
            distances_to_assigned = obs_distance_matrix[assigned_cluster]
        
            assinged_cluster_list.append(assigned_cluster)
            fb_distance_to_assinged_cluster_list.append(distances_to_assigned)
        
            #find best alternative:
        
            temp_bad = [] # best alternative distance
            temp_idx = []
        
            #for every feature
            for i in range(num_features):
            
                # best alternative: 
                best_alternative_distance = min(obs_distance_matrix[:,i]) # minimum distance to cluster
                max_distance = max(obs_distance_matrix[:,i])
                x = obs_distance_matrix[:,i].tolist() # nur um index zu bekommen --> zu welchen cluster gehört es
                idx_best_alternative = x.index(best_alternative_distance) # welches cluster
            
            
                #if the best alternative is the assigned cluster, we have to find the second best alternative
                if idx_best_alternative == assigned_cluster:
                    
                    # ensure second closest cluster is chosen as new min
                    x[idx_best_alternative] =  x[idx_best_alternative] + max_distance  
                    best_alternative_distance = min(x) 
                    idx_best_alternative = x.index(best_alternative_distance)
                    
                temp_bad.append(best_alternative_distance)
                temp_idx.append(idx_best_alternative)

            best_alterantive_list.append(temp_idx)
            fb_distance_to_best_alternative_list.append(temp_bad)     
            
        return np.array(fb_distance_to_assinged_cluster_list), np.array(fb_distance_to_best_alternative_list)
   
    def _calculate_pointwise_relevance(self) -> pd.DataFrame: # TODO: needs refactoring
        self._check_fitted()
        pointwise_scores = (self.fb_ba - self.fb_ac) / (self.fb_ba + self.fb_ac)  # type: ignore
        return (pd.DataFrame(pointwise_scores)
                .pipe(self._rename_feature_columns, self.num_features))

    def _calculate_cluster_relevance(self, pointwise_scores: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        return (pointwise_scores
                .pipe(self._rename_feature_columns, self.num_features)
                .assign(assigned_clusters=self.cluster_predictions)
                .groupby(["assigned_clusters"])
                .mean())

    def _calculate_global_relevance(self, pointwise_scores: pd.DataFrame) -> pd.Series:
        self._check_fitted()
        return pointwise_scores.mean()
    
    def fit(self):
        if not self.is_fitted:
            self.feature_wise_distance_matrix = self._calculate_feature_wise_distance_matrix()
            self.fb_ac , self.fb_ba = self._best_calc() # TODO this is very hard to test... --> needs refactoring
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


