import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

from cxplain.errors import NonExsitingXkmFlavourError, NotFittedError
from cxplain.xkm import ExplainedClustering, XkmExplainer, _get_xkm_flavour


@pytest.fixture()
def data():
    return np.array([[1, 2, 1, 2], [2, 3, 2, 3], [2, 2, 1, 2]])


@pytest.fixture()
def cluster_centers():
    return np.array([[1.5, 2.0, 1.0, 2.0], [2.0, 3.0, 2.0, 3.0]])


@pytest.fixture()
def predictions():
    return np.array([0, 1, 0])


@pytest.fixture()
def xkm(data, cluster_centers, predictions):
    return XkmExplainer(
        data=data,
        cluster_centers=cluster_centers,
        distance_metric="euclidean",
        flavour="next_best",
        cluster_predictions=predictions,
    )


@pytest.fixture()
def expected_explanation():
    col_names = ["R1", "R2", "R3", "R4"]
    pointwise_relevance = pd.DataFrame(
        np.array([[0.6, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, 1.0]]),
        columns=col_names,
    )
    cluster_relevance = pd.DataFrame(
        np.array([[-0.2, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]), columns=col_names
    )
    global_relevance = pd.Series(np.array([0.2, 1.0, 1.0, 1.0]), index=col_names)
    return ExplainedClustering(
        pointwise_relevance=pointwise_relevance,
        cluster_relevance=cluster_relevance,
        global_relevance=global_relevance,
    )


def test__calculate_feature_wise_distance_matrix(xkm):
    expected_distance_matrix = np.array(
        [
            [[0.25, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
            [[0.25, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.25, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]],
        ]
    )
    actual_distance_matrix = xkm._calculate_feature_wise_distance_matrix()
    np.testing.assert_allclose(expected_distance_matrix, actual_distance_matrix)


def test__best_calc(xkm, predictions):
    expected = (
        np.array([[0.25, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.25, 0.0, 0.0, 0.0]]),
        np.array([[1.0, 1.0, 1.0, 1.0], [0.25, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]]),
    )
    xkm.feature_wise_distance_matrix = xkm._calculate_feature_wise_distance_matrix()
    actual = xkm.flavour._best_calc(xkm.feature_wise_distance_matrix, predictions)
    np.testing.assert_allclose(expected, actual)


def test_explain(xkm, expected_explanation):
    with pytest.raises(NotFittedError):
        xkm.explain()

    actual_explanation = xkm.fit_explain()
    assert expected_explanation == actual_explanation


@pytest.fixture()
def iris_data():
    return load_iris().data


@pytest.fixture()
def iris_feature_names():
    return load_iris().feature_names


@pytest.fixture()
def iris_kmeans(iris_data):
    model = KMeans(n_clusters=3, random_state=0, n_init=10)
    model.fit(iris_data)
    return model


@pytest.fixture()
def iris_cluster_centers(iris_kmeans):
    return iris_kmeans.cluster_centers_


@pytest.fixture()
def iris_predictions(iris_kmeans, iris_data):
    return iris_kmeans.predict(iris_data)


@pytest.mark.parametrize(
    "flavour", ["next_best", "all", "within_scatter", "scatter_ratio"]
)
def test_get_xkm_flavour_all_supported_values(flavour):
    assert _get_xkm_flavour(flavour).__class__.__name__.startswith("Xkm")


def test_get_xkm_flavour_invalid_raises():
    with pytest.raises(NonExsitingXkmFlavourError):
        _get_xkm_flavour("does_not_exist")


@pytest.mark.parametrize(
    "flavour, expected_head5, expected_cluster, expected_global",
    [
        (
            "next_best",
            np.array(
                [
                    [0.97287154, 0.94453514, 0.99914246, 0.99722411],
                    [0.97784841, -0.94242865, 0.99914246, 0.99722411],
                    [0.87819783, -0.53029867, 0.99453038, 0.99722411],
                    [0.82266471, -0.98720829, 0.99965513, 0.99722411],
                    [0.99991143, 0.80701408, 0.99914246, 0.99722411],
                ]
            ),
            np.array(
                [
                    [0.32844918, 0.2462487, 0.71156379, 0.57821513],
                    [0.68629046, 0.22137365, 0.99286586, 0.98044763],
                    [0.53781171, 0.08489935, 0.75495688, 0.6210684],
                ]
            ),
            np.array([0.50076812, 0.19708185, 0.81632406, 0.72314879]),
        ),
        (
            "all",
            np.array(
                [
                    [0.99286253, 0.97931505, 0.99958546, 0.99873681],
                    [0.99300223, -1.18143351, 0.99958546, 0.99873681],
                    [0.95439812, 0.42642331, 0.99731552, 0.99873681],
                    [0.92855512, -0.39172338, 0.99983572, 0.99873681],
                    [0.9999745, 0.9139864, 0.99958546, 0.99873681],
                ]
            ),
            np.array(
                [
                    [0.79666695, 0.54296964, 0.93272109, 0.89589757],
                    [0.88701362, 0.39343256, 0.99660871, 0.99164587],
                    [0.83217766, 0.68656449, 0.96732584, 0.92075503],
                ]
            ),
            np.array([0.83577855, 0.52950131, 0.9627835, 0.93411089]),
        ),
        (
            "within_scatter",
            np.array(
                [
                    [0.5582, 0.7408, 0.8078, 0.8942],
                    [0.4382, -8.1592, 0.8078, 0.8942],
                    [-3.6818, -1.5992, -0.3122, 0.8942],
                    [-7.2418, -4.3792, 0.9278, 0.8942],
                    [0.9982, -0.4792, 0.8078, 0.8942],
                ]
            ),
            np.array(
                [
                    [-12.26983871, -4.35483871, -14.79741935, -4.39887097],
                    [-5.0882, -6.0408, -0.4778, 0.4558],
                    [-8.035, -2.11368421, -7.83263158, -1.89815789],
                ]
            ),
            np.array([-8.80313333, -4.34906667, -8.2598, -2.14713333]),
        ),
        (
            "scatter_ratio",
            np.array(
                [
                    [0.99717185, 0.99381849, 0.99984891, 0.99953479],
                    [0.99728199, -1.21693109, 0.99984891, 0.99953479],
                    [0.9823445, 0.80385583, 0.99902306, 0.99953479],
                    [0.97228857, 0.30062848, 0.99993998, 0.99953479],
                    [0.99999003, 0.97334365, 0.99984891, 0.99953479],
                ]
            ),
            np.array(
                [
                    [0.88802929, 0.43541407, 0.96699286, 0.94897413],
                    [0.94154548, 0.48341298, 0.99875308, 0.99681521],
                    [0.95192748, 0.91068155, 0.99165128, 0.97724176],
                ]
            ),
            np.array([0.92205556, 0.5718148, 0.9838264, 0.97208229]),
        ),
    ],
)
def test_fit_explain_all_flavours_iris_snapshots(
    flavour,
    expected_head5,
    expected_cluster,
    expected_global,
    iris_data,
    iris_cluster_centers,
    iris_predictions,
    iris_feature_names,
):
    explainer = XkmExplainer(
        data=iris_data,
        cluster_centers=iris_cluster_centers,
        distance_metric="euclidean",
        flavour=flavour,
        cluster_predictions=iris_predictions,
        feature_names=iris_feature_names,
    )

    with pytest.raises(NotFittedError):
        explainer.explain()

    explanation = explainer.fit_explain()

    assert explanation.pointwise_relevance_df.shape == (150, 4)
    assert explanation.cluster_relevance_df.shape == (3, 4)
    assert list(explanation.pointwise_relevance_df.columns) == iris_feature_names

    np.testing.assert_allclose(
        explanation.pointwise_relevance_df.head(5).to_numpy(),
        expected_head5,
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        explanation.cluster_relevance_df.to_numpy(),
        expected_cluster,
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        explanation.global_relevance_df.to_numpy(),
        expected_global,
        rtol=1e-6,
        atol=1e-8,
    )
