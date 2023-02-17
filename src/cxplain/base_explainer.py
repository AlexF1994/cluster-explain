from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from cxplain.errors import NonExistingRelevanceError, NotFittedError

# TODO incorporate feature names in relevance output 
# TODO support for different data types (categorical)
    

@dataclass()
class GlobalExplainedClustering:
    global_relevance: pd.Series
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GlobalExplainedClustering):
            return False
        global_equal = np.all(np.isclose(self.global_relevance.values, other.global_relevance.values))
        return global_equal
    
    def show_global_relevance(self): # TODO
        pass
    
    
@dataclass()
class ClusterExplainedClustering: 
    cluster_relevance: pd.DataFrame
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClusterExplainedClustering):
            return False
        cluster_equal = np.all(np.isclose(self.cluster_relevance.values, other.cluster_relevance.values))
        return cluster_equal
    
    def show_cluster_relevance(self): # TODO
        pass
    

@dataclass()
class PointwiseExplainedClustering:
    pointwise_relevance: pd.DataFrame
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PointwiseExplainedClustering):
            return False
        pointwise_equal = np.all(np.isclose(self.pointwise_relevance.values, other.pointwise_relevance.values))
        return pointwise_equal
    
    def show_pointwise_relevance(self): # TODO
        pass
 
 
class ExplainedClustering:
    def __init__(self, 
                 global_relevance: pd.Series,
                 pointwise_relevance: Optional[pd.DataFrame] = None,
                 cluster_relevance: Optional[pd.DataFrame] = None,
                 ):
        self._pointwise_relevance =  (PointwiseExplainedClustering(pointwise_relevance) 
                                      if isinstance(pointwise_relevance, pd.DataFrame)  
                                      else None)
        self._cluster_relevance = (ClusterExplainedClustering(cluster_relevance) 
                                   if isinstance(cluster_relevance, pd.DataFrame) 
                                   else None)
        self._global_relevance =  GlobalExplainedClustering(global_relevance)
        
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExplainedClustering):
            return False
        
        pointwise_equal = (self.pointwise_relevance == other.pointwise_relevance) if self.pointwise_relevance else True
        cluster_equal = (self.cluster_relevance == other.cluster_relevance) if self.cluster_relevance else True
        global_equal = (self.global_relevance == other.global_relevance)
        return pointwise_equal and cluster_equal and global_equal
        
    @property
    def pointwise_relevance(self)-> Optional[pd.DataFrame]:
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
    def _check_relevance_exists(relevance: Optional[Union[PointwiseExplainedClustering, 
                                                          ClusterExplainedClustering]] = None
                                ):
        if not (isinstance(relevance, PointwiseExplainedClustering) 
            or isinstance(relevance, ClusterExplainedClustering)):
            raise NonExistingRelevanceError("The specific relevance score doesn't exist for your explainer!")

    def show_pointwise_relevance(self): 
        self.pointwise_relevance.show_pointwise_relevance()

    def show_cluster_relevance(self): 
        self.cluster_relevance.show_cluster_relevance()

    def show_global_relevance(self):
        self.global_relevance.show_global_relevance()


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
            raise NotFittedError("You have to calculate the feature wise distance matrix and the best alternative"
            " and the best alternative distances first!")
            
    @staticmethod
    def _rename_feature_columns(df: pd.DataFrame, 
                                num_features: int, 
                                feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        if feature_names:
            assert num_features == len(feature_names)
        return df.rename({index_: feature_names[index_] if feature_names else f"R{index_ + 1}"
                          for index_ 
                          in range(num_features)},
                         axis=1
                         )