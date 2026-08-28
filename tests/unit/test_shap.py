import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

from cxplain.errors import NotFittedError
from cxplain.shap import ShapExplainer


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


def test_explain_raises_when_not_fitted(iris_data, iris_predictions):
    explainer = ShapExplainer(data=iris_data, cluster_predictions=iris_predictions)
    with pytest.raises(NotFittedError):
        explainer.explain()


def test_fit_explain_returns_expected_iris_snapshot(
    iris_data,
    iris_predictions,
    iris_feature_names,
):
    explainer = ShapExplainer(
        data=iris_data,
        cluster_predictions=iris_predictions,
        feature_names=iris_feature_names,
        random_state=0,
        n_estimators=50,
    )

    explanation = explainer.fit_explain()

    assert explanation.pointwise_relevance_df.shape == (150, 4)
    assert explanation.cluster_relevance_df.shape == (3, 4)
    assert list(explanation.pointwise_relevance_df.columns) == iris_feature_names

    expected_head5 = np.array(
        [
            [0.05220375, 0.0112709, 0.26421229, 0.33711306],
            [0.0558843, 0.00052822, 0.26327479, 0.34511269],
            [0.0558843, 0.00052822, 0.26327479, 0.34511269],
            [0.0558843, 0.00052822, 0.26327479, 0.34511269],
            [0.04591486, 0.01678201, 0.26499007, 0.33711306],
        ]
    )
    expected_cluster = np.array(
        [
            [0.06231613, 0.01238686, 0.32319705, 0.17442684],
            [0.04584608, 0.01007623, 0.26557333, 0.34250436],
            [0.1400036, 0.0151943, 0.38157062, 0.17135078],
        ]
    )
    expected_global = np.array([0.07650694, 0.01232787, 0.31877711, 0.22967341])

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


def test_relevant_shap_value_selection_shape(iris_predictions):
    rows = iris_predictions.shape[0]
    shap_values = np.zeros((rows, 4, 3))
    for i in range(rows):
        shap_values[i, :, iris_predictions[i]] = i + 1

    relevant = ShapExplainer._get_relevant_shap_values(shap_values, iris_predictions)
    assert relevant.shape == (rows, 4)
    np.testing.assert_allclose(relevant[0], np.array([1.0, 1.0, 1.0, 1.0]))
