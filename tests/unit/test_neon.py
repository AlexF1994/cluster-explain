import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

from cxplain.errors import NotFittedError
from cxplain.neon import NeonKMeansExplainer


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


def test_explain_raises_when_not_fitted(
    iris_data,
    iris_cluster_centers,
    iris_predictions,
):
    explainer = NeonKMeansExplainer(
        data=iris_data,
        cluster_centers=iris_cluster_centers,
        predictions=iris_predictions,
    )
    with pytest.raises(NotFittedError):
        explainer.explain()


def test_fit_initializes_networks_for_all_observations(
    iris_data,
    iris_cluster_centers,
    iris_predictions,
):
    explainer = NeonKMeansExplainer(
        data=iris_data,
        cluster_centers=iris_cluster_centers,
        predictions=iris_predictions,
    )
    explainer.fit()
    assert len(explainer.networks) == iris_data.shape[0]


def test_fit_explain_returns_expected_iris_snapshot(
    iris_data,
    iris_cluster_centers,
    iris_predictions,
    iris_feature_names,
):
    explainer = NeonKMeansExplainer(
        data=iris_data,
        cluster_centers=iris_cluster_centers,
        predictions=iris_predictions,
        feature_names=iris_feature_names,
    )

    explanation = explainer.fit_explain()

    assert explanation.pointwise_relevance_df.shape == (150, 4)
    assert explanation.cluster_relevance_df.shape == (3, 4)
    assert list(explanation.pointwise_relevance_df.columns) == iris_feature_names

    expected_head5 = np.array(
        [
            [3.84030905, 3.90558126, 3.58400297, 3.34829464],
            [3.60261353, 3.65371314, 3.35880143, 3.12970247],
            [4.03346919, 4.10808425, 3.74775656, 3.50054496],
            [3.59798687, 3.66363214, 3.37320111, 3.129197],
            [3.95325964, 4.02706374, 3.69097261, 3.4468457],
        ]
    )
    expected_cluster = np.array(
        [
            [1.08504956, 1.05153938, 1.15498647, 1.20115047],
            [3.66875733, 3.73120106, 3.43096869, 3.21972274],
            [0.49125074, 0.42940106, 0.59041221, 0.63170203],
        ]
    )
    expected_global = np.array([1.79585645, 1.78715157, 1.77062173, 1.72974762])

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


def test_beta_is_positive_and_finite(
    iris_data,
    iris_cluster_centers,
    iris_predictions,
):
    explainer = NeonKMeansExplainer(
        data=iris_data,
        cluster_centers=iris_cluster_centers,
        predictions=iris_predictions,
    ).fit()
    beta = explainer._get_beta()
    assert np.isfinite(beta)
    assert beta > 0
