from turtle import Shape
from typing import Any

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
import pandas as pd
from metrics import get_distance_metric
from nptyping import NDArray
from nptyping.typing_ import Floating, Int
from sklearn import datasets, metrics
from sklearn.cluster import KMeans
from errors import NotFittedError


class Xkm:

    "eXplainable k-medoids"
    
    def __init__(self, data: NDArray[Any, Floating], 
                 cluster_centers: NDArray[Any, Floating],
                 distance_metric: str, 
                 cluster_predictions: NDArray[Shape["*"], Int]
                 ):
        self.distance_metric = distance_metric
        self.cluster_centers =cluster_centers
        self.data = data
        self.predictions = cluster_predictions
        self.feature_wise_distance_matrix = None
        
    
    def _calculate_feature_wise_distance_matrix(self):
         
        centers = np.array([self.cluster_centers for observation_coordinates in self.data])

        # calculate the distance of every feature value of ever obs to every feature value in every cluster.
    
        feature_wise_distance_matrix = []
        distance_metric = get_distance_metric(self.distance_metric)

        feature_wise_distance_matrix = [distance_metric.calculate(cluster_centers, observation_ccordinates)
                                        for cluster_centers, observation_ccordinates 
                                        in zip(centers, self.data)]                
    
        self.feature_wise_distance_matrix = np.array(feature_wise_distance_matrix)
    
    def _best_calc(self):
        if not self.feature_wise_distance_matrix:
            raise NotFittedError("Please calculate the featurwise distance matrix first!")
        
        num_features = self.feature_wise_distance_matrix.shape[2]
    
        assinged_cluster_list = []
        fb_distance_to_assinged_cluster_list = []
    
        best_alterantive_list = []
        fb_distance_to_best_alternative_list = []
    
        #for every obs:
        for idx, e in enumerate(self.feature_wise_distance_matrix):
            #index of assinged cluster
            assigned_cluster = self.predictions[idx]
            #feature-wise distances of point to assigned cluster
            distances_to_assigned = e[assigned_cluster]
        
            assinged_cluster_list.append(assigned_cluster)
            fb_distance_to_assinged_cluster_list.append(distances_to_assigned)
        
            #find best alternative:
        
            temp_bad = []
            temp_idx = []
        
            #for every feature
            for i in range(num_features):
            
                # best alternative: 
                best_alternative_distance = min(e[:,i])
                x = e[:,i].tolist()
                idx_best_alternative = x.index(best_alternative_distance)
            
            
                #if the best alternative is the assigned cluster, we have to find the second best alternative
                if idx_best_alternative == assigned_cluster:
                
                    del x[idx_best_alternative]
                    best_alternative_distance = min(x)
                    idx_best_alternative = x.index(best_alternative_distance)
                    
                temp_bad.append(best_alternative_distance)
                temp_idx.append(idx_best_alternative)

            best_alterantive_list.append(temp_idx)
            fb_distance_to_best_alternative_list.append(temp_bad)     
            
        self.ac ,self.fb_ac ,self.ba, self.fb_ba = np.array(assinged_cluster_list), np.array(fb_distance_to_assinged_cluster_list), np.array(best_alterantive_list), np.array(fb_distance_to_best_alternative_list)
    
    def _calculate_pointwise_relevance(self):
        self.pointwise_scores = (self.fb_ba - self.fb_ac) / (self.fb_ba + self.fb_ac)   # type: ignore

    def _calculate_cluster_relevance(self):
        num_features = self.pointwise_scores.shape[1]
        self.cluster_relevances =  (pd.DataFrame(self.pointwise_scores)
                .rename({index_:f"R{index_ + 1}"
                        for index_ 
                        in range(num_features)},
                                    axis=1
                        )
                .assign(assigned_clusters=self.predictions)
                .groupby(["assigned_clusters"])
                .mean())

    def _calculate_global_relevance(self):
        self.global_relevances = {f"R_global_{i}": np.sum(self.pointwise_scores[:,i]) / len(self.pointwise_scores)
                                  for i in range(self.pointwise_scores.shape[1])} 
        
    def explain(self):
        self._calculate_feature_wise_distance_matrix()
        self._best_calc()
        self._calculate_pointwise_relevance()
        self._calculate_cluster_relevance()
        self._calculate_global_relevance()