from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cxplain.errors import NotFittedError


@dataclass()
class ExplainedClustering:
    pointwise_relevance: pd.DataFrame
    cluster_relevance: pd.DataFrame
    global_relevance: pd.Series
    # TODO
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExplainedClustering):
            return False
        
        pointwise_equal = np.all(np.isclose(self.pointwise_relevance.values, other.pointwise_relevance.values))
        cluster_equal = np.all(np.isclose(self.cluster_relevance.values, other.cluster_relevance.values))
        global_equal = np.all(np.isclose(self.global_relevance.values, other.global_relevance.values))
        return pointwise_equal and cluster_equal and global_equal
    
    def show_point_wise_relevance(self): # TODO
        pass 

    def show_cluster_relevance(self): # TODO
        pass

    def show_global_relevance(self): # TODO
        pass

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
    def _rename_feature_columns(df: pd.DataFrame, num_features: int) -> pd.DataFrame:
        return df.rename({index_:f"R{index_ + 1}"
                          for index_ 
                          in range(num_features)},
                         axis=1
                         )