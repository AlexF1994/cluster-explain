import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError as SklearnNotFittedError

from cxplain.errors import NotFittedError
from cxplain.exkmc import ExKMCExplainer


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


def test_exkmc_explain_raises_when_not_fitted(iris_data, iris_kmeans):
    explainer = ExKMCExplainer(data=iris_data, kmeans_fitted=iris_kmeans, k=3)
    with pytest.raises(NotFittedError):
        explainer.explain()


def test_exkmc_fit_explain_iris_global_snapshot(
    iris_data,
    iris_feature_names,
    iris_kmeans,
):
    explainer = ExKMCExplainer(
        data=iris_data,
        kmeans_fitted=iris_kmeans,
        feature_names=iris_feature_names,
        k=3,
    )
    explanation = explainer.fit_explain()

    expected_global = np.array([0.0, 0.0, 2.0, 0.0])
    assert list(explanation.global_relevance_df.index) == iris_feature_names
    np.testing.assert_allclose(
        explanation.global_relevance_df.to_numpy(),
        expected_global,
        rtol=1e-6,
        atol=1e-8,
    )


def test_exkmc_requires_fitted_kmeans(iris_data):
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    explainer = ExKMCExplainer(data=iris_data, kmeans_fitted=kmeans, k=3)
    with pytest.raises(SklearnNotFittedError):
        explainer.fit()
