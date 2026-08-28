from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class Dataset(ABC):
    def __init__(self, targets, features, feature_names, n_clusters=None):
        self.features = features
        self.feature_names = feature_names
        self.targets = targets
        self.n_obs = features.shape[0]
        self.n_features = features.shape[1]
        self.n_clusters = n_clusters if n_clusters else len(np.unique(targets))

    @classmethod
    @abstractmethod
    def _load_data(cls, path=None):
        pass

    @classmethod
    @abstractmethod
    def _clean_data(cls, data):
        pass

    @classmethod
    def load_and_clean_dataset(cls, path=None, n_clusters=None, **kwargs):
        data = cls._load_data(path, **kwargs)
        cleaned_data = cls._clean_data(data)
        features = cleaned_data.iloc[:, 1:]
        feature_names = list(features.columns)
        targets_df = cleaned_data.iloc[:, 0]
        return cls(targets_df.values, features.values, feature_names, n_clusters)  #

    def get_scree_plot(self, max_n_cluster=30):
        sse = {}
        for n_cluster in range(1, max_n_cluster):
            kmeans = KMeans(n_clusters=n_cluster, max_iter=1000, n_init="auto").fit(self.features)
            sse[n_cluster] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
        plt.figure().set_figwidth(15)
        plt.plot(list(sse.keys()), list(sse.values()))
        plt.xticks(list(range(1, max_n_cluster)))
        plt.xlabel("Number of cluster")
        plt.ylabel("SSE")
        plt.show()


class SkLearnDataset(Dataset):
    @classmethod
    def _clean_data(cls, data):
        return data

    @classmethod
    def load_and_clean_dataset(cls, n_clusters=None):
        data = cls._load_data()
        cleaned_data = cls._clean_data(data)
        return cls(
            cleaned_data.target,
            cleaned_data.data,
            cleaned_data.feature_names,
            n_clusters,
        )


class IrisDataset(SkLearnDataset):
    @classmethod
    def _load_data(cls):
        return datasets.load_iris()


class WineDataset(SkLearnDataset):
    @classmethod
    def _load_data(cls):
        return datasets.load_wine()

    @classmethod
    def _clean_data(cls, data):
        data.data = MinMaxScaler().fit_transform(data.data)
        return data


class WholeSaleDataset(Dataset):
    @classmethod
    def _load_data(cls, path, **kwargs):
        return pd.read_csv(path, **kwargs)

    @classmethod
    def _clean_data(cls, data):
        features = data.drop(["Channel", "Region"], axis=1)
        targets = data["Region"]
        feature_names = list(features.columns)
        features_scaled = pd.DataFrame(MinMaxScaler().fit_transform(features), columns=feature_names)
        return pd.concat(
            [targets.reset_index(drop=True), features_scaled.reset_index(drop=True)],
            axis=1,
        )


class LiveSellersDataset(Dataset):
    @classmethod
    def _load_data(cls, path=None, **kwargs):
        return pd.read_csv(path, **kwargs)

    @classmethod
    def _clean_data(cls, data):
        data_cleaned = data.iloc[:, :-4]
        targets = data_cleaned["status_type"]
        targets.map({"video": 0, "photo": 1, "link": 2, "status": 3})
        features = data_cleaned.drop(["status_id", "status_type", "status_published"], axis=1)
        feature_names = list(features.columns)
        features_scaled = pd.DataFrame(MinMaxScaler().fit_transform(features), columns=feature_names)
        return pd.concat(
            [targets.reset_index(drop=True), features_scaled.reset_index(drop=True)],
            axis=1,
        )


class MissingTargetsMixin:
    @classmethod
    def load_and_clean_dataset(cls, n_clusters, path=None, **kwargs):
        data = cls._load_data(path, **kwargs)
        features = cls._clean_data(data)
        feature_names = list(features.columns)
        return cls(None, features.values, feature_names, n_clusters)


class BuddyMoveDataset(MissingTargetsMixin, Dataset):
    @classmethod
    def _load_data(cls, path=None, **kwargs):
        return pd.read_csv(path, **kwargs)

    @classmethod
    def _clean_data(cls, data):
        features = data.drop("User Id", axis=1)
        feature_names = list(features.columns)
        return pd.DataFrame(MinMaxScaler().fit_transform(features), columns=feature_names)


class SyntheticDataset(MissingTargetsMixin, Dataset):
    @classmethod
    def _load_data(cls, path=None, **kwargs):
        return pd.read_csv(path, names=["f_1", "f_2"], header=None, sep=r"\s+", **kwargs)

    @classmethod
    def _clean_data(cls, data):
        features = data.copy()
        feature_names = list(features.columns)
        return pd.DataFrame(MinMaxScaler().fit_transform(features), columns=feature_names)


class MallCustomersDataset(MissingTargetsMixin, Dataset):
    @classmethod
    def _load_data(cls, path=None, **kwargs):
        return pd.read_csv(path, **kwargs)

    @classmethod
    def _clean_data(cls, data):
        data = data.drop(["Gender"], axis=1)
        # data["Gender"] = data["Gender"].replace({"Male": 0, "Female": 1})
        feature_names = list(data.columns)
        return pd.DataFrame(
            MinMaxScaler().fit_transform(data),
            columns=feature_names,
        )
