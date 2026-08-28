import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

from cxplain.errors import NotFittedError
from cxplain.tree import DecisionTreeExplainer, RandomForestExplainer


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
def iris_predictions(iris_kmeans, iris_data):
    return iris_kmeans.predict(iris_data)


def test_decision_tree_explain_raises_when_not_fitted(iris_data, iris_predictions):
    explainer = DecisionTreeExplainer(
        data=iris_data, cluster_predictions=iris_predictions
    )
    with pytest.raises(NotFittedError):
        explainer.explain()


def test_random_forest_explain_raises_when_not_fitted(iris_data, iris_predictions):
    explainer = RandomForestExplainer(
        data=iris_data, cluster_predictions=iris_predictions
    )
    with pytest.raises(NotFittedError):
        explainer.explain()


def test_decision_tree_fit_explain_iris_global_snapshot(
    iris_data,
    iris_predictions,
    iris_feature_names,
):
    explainer = DecisionTreeExplainer(
        data=iris_data,
        cluster_predictions=iris_predictions,
        feature_names=iris_feature_names,
        random_state=0,
    )
    explanation = explainer.fit_explain()

    expected_global = np.array([0.01616162, 0.0, 0.29797172, 0.33973333])
    assert list(explanation.global_relevance_df.index) == iris_feature_names
    np.testing.assert_allclose(
        explanation.global_relevance_df.to_numpy(),
        expected_global,
        rtol=1e-6,
        atol=1e-8,
    )


def test_random_forest_fit_explain_iris_global_snapshot(
    iris_data,
    iris_predictions,
    iris_feature_names,
):
    explainer = RandomForestExplainer(
        data=iris_data,
        cluster_predictions=iris_predictions,
        feature_names=iris_feature_names,
        random_state=0,
        n_estimators=100,
    )
    explanation = explainer.fit_explain()

    expected_global = np.array([0.16115907, 0.02785407, 0.49831773, 0.31266913])
    assert list(explanation.global_relevance_df.index) == iris_feature_names
    np.testing.assert_allclose(
        explanation.global_relevance_df.to_numpy(),
        expected_global,
        rtol=1e-6,
        atol=1e-8,
    )
