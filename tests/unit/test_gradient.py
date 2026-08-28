import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

from cxplain.gradient import GradientExplainer


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
    "enable_abs, expected_cluster, expected_global",
    [
        (
            True,
            np.array(
                [
                    [0.73891779, 0.47450572, 0.82185224, 0.4635796],
                    [0.54144, 0.57472, 0.26304, 0.16512],
                    [0.81052632, 0.42880886, 0.76066482, 0.4565097],
                ]
            ),
            np.array([0.69123269, 0.49633394, 0.62008068, 0.36230203]),
        ),
        (
            False,
            np.array(
                [
                    [0.0, -0.0, -0.0, 0.0],
                    [-0.0, 0.0, -0.0, -0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            np.array([0.0, 0.0, 0.0, -0.0]),
        ),
    ],
)
def test_explain_iris_snapshot(
    enable_abs,
    expected_cluster,
    expected_global,
    iris_data,
    iris_cluster_centers,
    iris_predictions,
    iris_feature_names,
):
    explainer = GradientExplainer(
        data=iris_data,
        cluster_centers=iris_cluster_centers,
        cluster_predictions=iris_predictions,
        enable_abs_calculation=enable_abs,
        feature_names=iris_feature_names,
    )

    explanation = explainer.explain()
    assert explanation.pointwise_relevance_df.shape == (150, 4)
    assert explanation.cluster_relevance_df.shape == (3, 4)
    assert list(explanation.pointwise_relevance_df.columns) == iris_feature_names

    expected_head5 = np.array(
        [
            [0.188, 0.144, -0.124, -0.092],
            [-0.212, -0.856, -0.124, -0.092],
            [-0.612, -0.456, -0.324, -0.092],
            [-0.812, -0.656, 0.076, -0.092],
            [-0.012, 0.344, -0.124, -0.092],
        ]
    )
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


def test_explain_without_explicit_fit(
    iris_data,
    iris_cluster_centers,
    iris_predictions,
):
    explainer = GradientExplainer(
        data=iris_data,
        cluster_centers=iris_cluster_centers,
        cluster_predictions=iris_predictions,
    )
    explanation = explainer.explain()
    assert explanation.pointwise_relevance_df.shape == (150, 4)
