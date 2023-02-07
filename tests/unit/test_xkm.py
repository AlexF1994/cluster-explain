import numpy as np
import pandas as pd
import pytest

from cxplain.xkm import ExplainedClustering, NotFittedError, Xkm


@pytest.fixture()
def data():
    return np.array([[1, 2, 1, 2],
                     [2, 3, 2, 3],
                     [2, 2, 1, 2]])
    
@pytest.fixture()
def cluster_centers():
    return np.array([[1.5, 2. , 1. , 2. ],
                     [2. , 3. , 2. , 3. ]])
    
@pytest.fixture()
def predictions():
    return np.array([0, 1, 0])

@pytest.fixture()
def xkm(data, cluster_centers, predictions):
    return Xkm(data=data,
               cluster_centers=cluster_centers,
               distance_metric="euclidean",
               cluster_predictions=predictions)
    
@pytest.fixture()
def expected_explanation():
    col_names = ["R1", "R2", "R3", "R4"]
    pointwise_relevance = pd.DataFrame(np.array([[ 0.6,  1. ,  1. ,  1. ],
                                                 [ 1. ,  1. ,  1. ,  1. ],
                                                 [-1. ,  1. ,  1. ,  1. ]]),
                                       columns=col_names)
    cluster_relevance = pd.DataFrame(np.array([[-0.2,  1. ,  1. ,  1. ],
                                               [ 1. ,  1. ,  1. ,  1. ]]),
                                       columns=col_names)
    global_relevance = pd.Series(np.array([0.2, 1. , 1. , 1. ]),
                                       index=col_names)
    return ExplainedClustering(pointwise_relevance=pointwise_relevance,
                               cluster_relevance=cluster_relevance,
                               global_relevance=global_relevance)


def test__calculate_feature_wise_distance_matrix(xkm):
    expected_distance_matrix = np.array([[[0.25, 0.  , 0.  , 0.  ],
                                          [1.  , 1.  , 1.  , 1.  ]],                                                                           
                                         [[0.25, 1.  , 1.  , 1.  ],
                                          [0.  , 0.  , 0.  , 0.  ]],
                                         [[0.25, 0.  , 0.  , 0.  ],
                                          [0.  , 1.  , 1.  , 1.  ]]])
    actual_distance_matrix = xkm._calculate_feature_wise_distance_matrix()
    np.testing.assert_allclose(expected_distance_matrix, actual_distance_matrix)
    
    
def test__best_calc(xkm):
    expected = (np.array([[0.25, 0.  , 0.  , 0.  ],
                          [0.  , 0.  , 0.  , 0.  ],
                          [0.25, 0.  , 0.  , 0.  ]]),
                np.array([[1.  , 1.  , 1.  , 1.  ],
                          [0.25, 1.  , 1.  , 1.  ],
                          [0.  , 1.  , 1.  , 1.  ]]))
    xkm.feature_wise_distance_matrix = xkm._calculate_feature_wise_distance_matrix()
    actual = xkm._best_calc()
    np.testing.assert_allclose(expected, actual)
    
    
def test_explain(xkm, expected_explanation):    
    with pytest.raises(NotFittedError):
        xkm.explain()
        
    actual_explanation = xkm.fit_explain()
    assert expected_explanation == actual_explanation