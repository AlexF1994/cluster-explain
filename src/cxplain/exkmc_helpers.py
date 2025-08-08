"""Utilities for explainable K-means clustering: tree construction and split search.

This module implements a light rework of the ExKMC package from https://github.com/navefr/ExKMC/tree/master
as the repo isn't developed anymore.
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cluster import KMeans

# the following is nearly 1:1 copy of https://github.com/navefr/ExKMC/tree/master
# I copied it as the original repo isn't furtehr developed anymore

BASE_TREE = ["IMM", "NONE"]

LEAF_DATA_KEY_X_DATA = "X_DATA_KEY"
LEAF_DATA_KEY_Y = "Y_KEY"
LEAF_DATA_KEY_X_CENTER_DOT = "X_CENTER_DOT"
LEAF_DATA_KEY_SPLITTER = "SPLITTER_KEY"

__all__ = [
    "Tree",
    "Node",
    "convert_input",
    "get_min_mistakes_cut",
    "get_min_surrogate_cut",
    "IMM_Cut",
    "Surrogate_Cut",
    "Surrogate_Split",
    "LeafData",
]


class Tree:
    """
    Explainable K-means tree for clustering explanations.

    This class implements a decision tree that approximates K-means clustering and provides
    feature importance scores based on the tree structure. The tree is built using the
    IMM (Iterative Mistake Minimization) algorithm.

    Attributes:
        k (int): Number of clusters.
        max_leaves (int): Maximum number of leaves in the tree.
        verbose (int): Verbosity level for logging.
        light (bool): If True, does not store input examples in leaves to save memory.
        base_tree (str): Base tree construction method, either "IMM" or "NONE".
        n_jobs (int): Number of parallel jobs for computation.
        random_state (int | None): Random state for reproducibility.
        tree (Node | None): Root node of the decision tree.
        _leaves_data (dict): Data associated with each leaf node.
        _feature_importance (NDArray | None): Feature importance scores.

    Methods:
        - fit(self, x_data, kmeans=None) -> Tree: Builds a threshold tree.
        - predict(self, x_data) -> NDArray: Predicts cluster assignments.
        - score(self, x_data) -> float: Returns the K-means cost.
        - surrogate_score(self, x_data) -> float: Returns the K-means surrogate cost.
    """

    def __init__(
        self,
        k: int,
        max_leaves: Optional[int] = None,
        verbose: int = 0,
        light: bool = True,
        base_tree: str = "IMM",
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize the explainable K-means tree.

        Args:
            k: Number of clusters.
            max_leaves: Maximum number of leaves. If None, defaults to k.
            verbose: Verbosity level for logging.
            light: If True, does not store input examples in leaves.
            base_tree: Base tree construction method ("IMM" or "NONE").
            n_jobs: Number of parallel jobs.
            random_state: Random state for reproducibility.

        Raises:
            ValueError: If max_leaves < k or base_tree is not supported.
        """
        self.k: int = k
        self.tree: Optional[Node] = None
        # refine type for leaf storage
        self._leaves_data: Dict[Node, LeafData] = {}
        self.max_leaves: int = k if max_leaves is None else max_leaves
        self.random_state: Optional[int] = random_state
        if self.max_leaves < k:
            raise ValueError(
                f"max_leaves must be greater or equal to k [{self.max_leaves} < {k}]"
            )
        self.verbose: int = verbose
        self.light: bool = light
        if base_tree not in BASE_TREE:
            raise ValueError(f"{base_tree} is not a supported base tree")
        self.base_tree: str = base_tree
        self.n_jobs: int = n_jobs if n_jobs is not None else 1
        self._feature_importance: Optional[NDArray[np.floating]] = None
        self.all_centers: NDArray[np.floating] | None = None

    def _build_tree(
        self,
        x_data: NDArray[np.floating],
        y: NDArray[np.integer],
        valid_centers: NDArray[np.integer],
        valid_cols: NDArray[np.integer],
    ) -> "Node":
        """Build a decision tree recursively.

        Args:
            x_data: Input samples of shape (n_samples, n_features).
            y: Cluster assignments for the input samples of shape (n_samples,).
            valid_centers: 0/1 mask of shape (n_centers,) for center candidates.
            valid_cols: 0/1 mask of shape (n_features,) for usable features.

        Returns:
            Root node of the created tree.
        """
        if self.verbose > 1:
            print(f"build node (samples={x_data.shape[0]})")
        node = Node()
        if x_data.shape[0] == 0:
            node.value = 0
            return node
        elif valid_centers.sum() == 1:
            node.value = int(np.argmax(valid_centers))
            return node
        else:
            if np.unique(y).shape[0] == 1:
                node.value = int(y[0])
                return node
            else:
                # Verify data type is float64/int32 prior to algorithm call
                x_data = x_data.astype(np.float64, copy=False)
                y = y.astype(np.int32, copy=False)
                assert self.all_centers is not None
                self.all_centers = self.all_centers.astype(np.float64, copy=False)
                valid_centers = valid_centers.astype(np.int32, copy=False)
                valid_cols = valid_cols.astype(np.int32, copy=False)

                cut = get_min_mistakes_cut(
                    x_data, y, self.all_centers, valid_centers, valid_cols, self.n_jobs
                )

                if cut is None:
                    node.value = int(np.argmax(valid_centers))
                else:
                    col = cut.col
                    threshold = cut.threshold
                    node.set_condition(col, threshold)

                    left_data_mask = x_data[:, col] <= threshold
                    matching_centers_mask = self.all_centers[:, col][y] <= threshold
                    mistakes_mask = left_data_mask != matching_centers_mask

                    left_valid_centers_mask = (
                        self.all_centers[valid_centers.astype(bool), col] <= threshold
                    )
                    left_valid_centers = np.zeros(valid_centers.shape, dtype=np.int32)
                    left_valid_centers[valid_centers.astype(bool)] = (
                        left_valid_centers_mask
                    )
                    right_valid_centers = np.zeros(valid_centers.shape, dtype=np.int32)
                    right_valid_centers[valid_centers.astype(bool)] = (
                        ~left_valid_centers_mask
                    )

                    node.left = self._build_tree(
                        x_data[left_data_mask & ~mistakes_mask],
                        y[left_data_mask & ~mistakes_mask],
                        left_valid_centers,
                        valid_cols,
                    )
                    node.right = self._build_tree(
                        x_data[~left_data_mask & ~mistakes_mask],
                        y[~left_data_mask & ~mistakes_mask],
                        right_valid_centers,
                        valid_cols,
                    )

                return node

    def fit(
        self, x_data: NDArray[np.floating], kmeans: Optional[KMeans] = None
    ) -> "Tree":
        """Build a threshold tree from training data and a K-means model.

        If no model is provided, a new KMeans is trained with n_clusters=k.

        Args:
            x_data: Training input samples.
            kmeans: Optional pre-trained KMeans model.

        Returns:
            The fitted tree instance (self).
        """
        x_data = convert_input(x_data)

        if kmeans is None:
            if self.verbose > 0:
                print(f"Finding {self.k}-means")
            # Use modern sklearn API to avoid deprecation warnings
            kmeans = KMeans(
                n_clusters=self.k,
                random_state=self.random_state,
                n_init="auto",
                max_iter=40,
            )
            kmeans.fit(x_data)
        else:
            if getattr(kmeans, "n_clusters", None) != self.k:
                raise ValueError(
                    f"Provided KMeans has n_clusters={getattr(kmeans, 'n_clusters', None)}, expected k={self.k}"
                )

        y = np.array(kmeans.predict(x_data), dtype=np.int32)

        self.all_centers = np.array(kmeans.cluster_centers_, dtype=np.float64)

        if self.base_tree == "IMM":
            self.tree = self._build_tree(
                x_data,
                y,
                np.ones(self.all_centers.shape[0], dtype=np.int32),
                np.ones(self.all_centers.shape[1], dtype=np.int32),
            )
            leaves = self.k
        else:
            self.tree = Node()
            self.tree.value = 0
            leaves = 1

        if self.max_leaves > leaves:
            self.__gather_leaves_data__(self.tree, x_data, y)
            all_centers_norm_sqr = (
                np.linalg.norm(self.all_centers, axis=1) ** 2
            ).astype(np.float64, copy=False)
            self.__expand_tree__(leaves, all_centers_norm_sqr)
            if self.light:
                self._leaves_data = {}

        self._feature_importance = np.zeros(x_data.shape[1])
        self.__fill_stats__(self.tree, x_data, y)

        return self

    def fit_predict(
        self, x_data: NDArray[np.floating], kmeans: Optional[KMeans] = None
    ) -> NDArray[np.int32]:
        """Fit the tree and return predicted clusters for x_data.

        Args:
            x_data: The training input samples.
            kmeans: Optional pre-trained KMeans model.

        Returns:
            Array of predicted cluster assignments.
        """
        self.fit(x_data, kmeans)
        return self.predict(x_data)

    def predict(self, x_data: NDArray[np.floating]) -> NDArray[np.int32]:
        """Predict cluster assignments for new data points.

        Args:
            x_data: The input samples to predict.

        Returns:
            Array of predicted cluster assignments as int32.
        """
        x_data = convert_input(x_data)
        if self.tree is None:
            raise RuntimeError("Tree is not fitted. Call fit() before predict().")
        return self._predict_subtree(self.tree, x_data)

    def _predict_subtree(
        self, node: "Node", x_data: NDArray[np.floating]
    ) -> NDArray[np.int32]:
        """Recursively predict cluster assignments for the subtree rooted at node."""
        if node.is_leaf():
            assert node.value is not None
            return np.full(x_data.shape[0], int(node.value), dtype=np.int32)
        else:
            ans = np.zeros(x_data.shape[0], dtype=np.int32)
            left_mask = x_data[:, node.feature] <= node.value  # type: ignore[index]
            ans[left_mask] = self._predict_subtree(node.left, x_data[left_mask])  # type: ignore[arg-type]
            ans[~left_mask] = self._predict_subtree(node.right, x_data[~left_mask])  # type: ignore[arg-type]
            return ans

    def score(self, x_data: NDArray[np.floating]) -> float:
        """Return the K-means cost for the given data.

        The K-means cost is the sum of squared distances of each point to the mean
        of points associated with its assigned cluster.

        Args:
            x_data: The input samples.

        Returns:
            The K-means cost of the data.
        """
        x_data = convert_input(x_data)
        clusters = self.predict(x_data)
        cost = 0.0
        for c in range(self.k):
            cluster_data = x_data[clusters == c, :]
            if cluster_data.shape[0] > 0:
                center = cluster_data.mean(axis=0)
                # Frobenius norm squared equals sum of per-point squared distances
                cost += float(np.linalg.norm(cluster_data - center) ** 2)
        return cost

    def surrogate_score(self, x_data: NDArray[np.floating]) -> float:
        """Return the K-means surrogate cost for the given data.

        The surrogate cost is the sum of squared distances to the closest center
        from the K-means model used in fit.

        Args:
            x_data: The input samples.

        Returns:
            The K-means surrogate cost of the data.
        """
        x_data = convert_input(x_data)
        clusters = self.predict(x_data)
        cost = 0.0
        assert self.all_centers is not None
        for c in range(self.k):
            cluster_data = x_data[clusters == c, :]
            if cluster_data.shape[0] > 0:
                center = self.all_centers[c]
                cost += float(np.linalg.norm(cluster_data - center) ** 2)
        return cost

    def _size(self) -> int:
        """Return the number of nodes in the threshold tree."""
        return self.__size__(self.tree)

    def __size__(self, node: Optional["Node"]) -> int:
        """Return the number of nodes in the subtree rooted by the given node."""
        if node is None:
            return 0
        else:
            sl = self.__size__(node.left)
            sr = self.__size__(node.right)
            return 1 + sl + sr

    def _max_depth(self) -> int:
        """Return the depth of the threshold tree."""
        return self.__max_depth__(self.tree)

    def __max_depth__(self, node: Optional["Node"]) -> int:
        """Return the depth of the subtree rooted by the given node."""
        if node is None:
            return -1
        else:
            dl = self.__max_depth__(node.left)
            dr = self.__max_depth__(node.right)
            return 1 + max(dl, dr)

    def __expand_tree__(
        self, size: int, all_centers_norm_sqr: NDArray[np.floating]
    ) -> None:
        """Greedily expand the tree by splitting leaves to reduce surrogate cost."""
        if size < self.max_leaves:
            if self.verbose > 1:
                print(f"expand tree. size {size}/{self.max_leaves}")

            best_splitter: Optional[Surrogate_Split] = None
            leaf_to_split: Optional[Node] = None
            leaf_count = 1
            for leaf in self._leaves_data:
                if self.verbose > 1:
                    print(
                        f"-- expand leaf. {leaf_count}/{len(self._leaves_data)} "
                        f"(samples={self._leaves_data[leaf].x_data.shape[0]})"
                    )
                if self._leaves_data[leaf].splitter is None:
                    self._leaves_data[leaf].splitter = self.__expand_leaf__(
                        leaf, all_centers_norm_sqr
                    )
                leaf_splitter: Optional[Surrogate_Split] = self._leaves_data[
                    leaf
                ].splitter
                if leaf_splitter is not None:
                    if (
                        best_splitter is None
                        or leaf_splitter.cost_gain < best_splitter.cost_gain
                    ):
                        best_splitter = leaf_splitter
                        leaf_to_split = leaf
                leaf_count += 1
            if best_splitter is not None and leaf_to_split is not None:
                col = best_splitter.col
                threshold = best_splitter.threshold
                self.__split_leaf__(
                    leaf_to_split,
                    col,
                    threshold,
                    best_splitter.center_left,
                    best_splitter.center_right,
                )

                assert (
                    leaf_to_split.left is not None and leaf_to_split.right is not None
                )
                X = self._leaves_data[leaf_to_split].x_data
                y = self._leaves_data[leaf_to_split].y
                X_center_dot = self._leaves_data[leaf_to_split].x_center_dot
                left_mask = X[:, col] <= threshold

                del self._leaves_data[leaf_to_split]

                self._leaves_data[leaf_to_split.left] = LeafData(  # type: ignore[index]
                    x_data=X[left_mask],
                    y=y[left_mask],
                    x_center_dot=X_center_dot[left_mask],
                )
                self._leaves_data[leaf_to_split.right] = LeafData(  # type: ignore[index]
                    x_data=X[~left_mask],
                    y=y[~left_mask],
                    x_center_dot=X_center_dot[~left_mask],
                )
                self.__expand_tree__(size + 1, all_centers_norm_sqr)

    def __gather_leaves_data__(
        self, node: "Node", x_data: NDArray[np.floating], y: NDArray[np.integer]
    ) -> None:
        """Collect data per leaf to enable greedy expansion with surrogate splits."""
        if node.is_leaf():
            assert self.all_centers is not None
            self._leaves_data[node] = LeafData(
                x_data=x_data,
                y=y.astype(np.int32, copy=False),
                x_center_dot=np.dot(x_data, self.all_centers.T).astype(
                    np.float64, copy=False
                ),
            )
        else:
            left_mask = x_data[:, node.feature] <= node.value  # type: ignore[index]
            self.__gather_leaves_data__(node.left, x_data[left_mask], y[left_mask])  # type: ignore[arg-type]
            self.__gather_leaves_data__(node.right, x_data[~left_mask], y[~left_mask])  # type: ignore[arg-type]

    def __expand_leaf__(
        self, leaf: "Node", all_centers_norm_sqr: NDArray[np.floating]
    ) -> Optional["Surrogate_Split"]:
        """Find the best surrogate split for a given leaf, if any."""
        leaf_data = self._leaves_data[leaf]
        mistakes_counter = Counter([curr_y for curr_y in leaf_data.y if curr_y != leaf.value])  # type: ignore[operator]
        if len(mistakes_counter) == 0:
            return None

        # Verify data type is float64 prior to computation
        X = leaf_data.x_data.astype(np.float64, copy=False)
        X_center_dot = leaf_data.x_center_dot.astype(np.float64, copy=False)
        all_centers_norm_sqr = all_centers_norm_sqr.astype(np.float64, copy=False)

        min_cut = get_min_surrogate_cut(
            X, X_center_dot, X_center_dot.sum(axis=0), all_centers_norm_sqr, self.n_jobs
        )

        if min_cut is not None:
            pre_split_cost = self.__get_leaf_pre_split_cost__(
                X_center_dot, all_centers_norm_sqr
            )
            cost_gain = float(min_cut.cost - pre_split_cost)
            return Surrogate_Split(
                col=int(min_cut.col),
                threshold=float(min_cut.threshold),
                cost_gain=cost_gain,
                center_left=int(min_cut.center_left),
                center_right=int(min_cut.center_right),
            )
        else:
            return None

    def __get_leaf_pre_split_cost__(
        self,
        X_center_dot: NDArray[np.floating],
        all_centers_norm_sqr: NDArray[np.floating],
    ) -> float:
        """Compute the best (minimal) surrogate cost for a leaf before splitting."""
        n = X_center_dot.shape[0]

        cost_per_center = (n * all_centers_norm_sqr) - 2 * X_center_dot.sum(axis=0)
        best_center = int(cost_per_center.argmin())
        return float(cost_per_center[best_center])

    def __split_leaf__(
        self,
        leaf: "Node",
        feature: int,
        value: float,
        left_cluster: int,
        right_cluster: int,
    ) -> None:
        """Turn a leaf into an internal node by setting its split and children."""
        leaf.feature = int(feature)
        leaf.value = float(value)

        leaf.left = Node()
        leaf.left.value = int(left_cluster)

        leaf.right = Node()
        leaf.right.value = int(right_cluster)

    def __fill_stats__(
        self, node: "Node", x_data: NDArray[np.floating], y: NDArray[np.integer]
    ) -> None:
        """Fill sample counts, mistakes, and feature importance statistics recursively."""
        node.samples = int(x_data.shape[0])
        if not node.is_leaf():
            assert self._feature_importance is not None
            self._feature_importance[node.feature] += 1  # type: ignore[index]
            left_mask = x_data[:, node.feature] <= node.value  # type: ignore[index]
            self.__fill_stats__(node.left, x_data[left_mask], y[left_mask])  # type: ignore[arg-type]
            self.__fill_stats__(node.right, x_data[~left_mask], y[~left_mask])  # type: ignore[arg-type]
        else:
            node.mistakes = int(
                len([cluster for cluster in y if cluster != node.value])
            )

    @property
    def feature_importance(self) -> NDArray[np.floating]:
        """Feature importance as split counts per feature."""
        assert self._feature_importance is not None
        return self._feature_importance


class Node:
    """
    Node class for the decision tree structure.

    Each node represents either a decision point (with a feature and threshold)
    or a leaf node (with a cluster assignment).

    Attributes:
        feature (int | None): Index of the feature used for splitting (None for leaf nodes).
        value (float | None): Threshold value for splitting or cluster assignment for leaf nodes.
        samples (int | None): Number of samples in this node.
        mistakes (int | None): Number of misclassified samples in this node.
        left (Node | None): Left child node (None for leaf nodes).
        right (Node | None): Right child node (None for leaf nodes).
    """

    __slots__ = ("feature", "value", "samples", "mistakes", "left", "right")

    def __init__(self) -> None:
        """Initialize a new node."""
        self.feature: Optional[int] = None
        self.value: Optional[float] = None
        self.samples: Optional[int] = None
        self.mistakes: Optional[int] = None
        self.left: Optional["Node"] = None
        self.right: Optional["Node"] = None

    def is_leaf(self) -> bool:
        """Check if this node is a leaf node."""
        return (self.left is None) and (self.right is None)

    def set_condition(self, feature: int, value: float) -> None:
        """Set the splitting condition for this node.

        Args:
            feature: Index of the feature to split on.
            value: Threshold value for the split.
        """
        self.feature = int(feature)
        self.value = float(value)


@dataclass(slots=True)
class LeafData:
    """Per-leaf cache used to greedily expand the tree with surrogate splits."""

    x_data: NDArray[np.floating]
    y: NDArray[np.int32]
    x_center_dot: NDArray[np.floating]
    splitter: Optional["Surrogate_Split"] = None


def convert_input(data: object) -> NDArray[np.floating]:
    """Convert input data to a 2D numpy array of dtype float64.

    Supports Python lists, numpy arrays, and pandas DataFrames.

    Args:
        data: Input data in various formats.

    Returns:
        Converted data as numpy array with float64 dtype.

    Raises:
        TypeError: If the input type is not supported.
    """
    if isinstance(data, list):
        data = np.array(data, dtype=np.float64)
    elif isinstance(data, np.ndarray):
        data = data.astype(np.float64, copy=False)
    elif isinstance(data, pd.DataFrame):
        data = data.values.astype(np.float64, copy=False)
    else:
        raise TypeError(f"{type(data)} is not supported type")
    # Ensure 2D
    if data.ndim != 2:
        raise ValueError("Input data must be 2-dimensional (n_samples, n_features).")
    return data


INF = float("inf")


@dataclass(frozen=True, slots=True)
class IMM_Cut:
    """A cut defined by a feature (column) and a threshold (IMM algorithm)."""

    col: int
    threshold: float


@dataclass(frozen=True, slots=True)
class Surrogate_Cut:
    """A surrogate cut with its total cost and the best centers on both sides."""

    col: int
    threshold: float
    cost: float
    center_left: int
    center_right: int


@dataclass(frozen=True, slots=True)
class Surrogate_Split:
    """A surrogate split enriched with cost gain for greedy leaf expansion."""

    col: int
    threshold: float
    cost_gain: float
    center_left: int
    center_right: int


def get_min_mistakes_cut(
    X: NDArray[np.floating],
    y: NDArray[np.integer],
    centers: NDArray[np.floating],
    valid_centers: NDArray[np.integer],
    valid_cols: NDArray[np.integer],
    njobs: Optional[int] = None,
) -> Optional[IMM_Cut]:
    """
    Find the feature and threshold that minimize the number of label mistakes.

    A mistake occurs when, given a threshold t on column col, a point x is placed
    to the left (x[col] <= t) while its corresponding center's value on that column
    is to the right (centers[y_i, col] >= t), or vice versa, following the original
    algorithm's semantics.

    Args:
        X: Array of shape (n_samples, n_features).
        y: Array of shape (n_samples,) with integer center indices in [0, n_centers).
        centers: Array of shape (n_centers, n_features).
        valid_centers: Array of shape (n_centers,) with 0/1 flags for valid centers.
        valid_cols: Array of shape (n_features,) with 0/1 flags for usable features.
        njobs: Ignored. Present for API compatibility.

    Returns:
        IMM_Cut if a valid cut is found, else None.

    Raises:
        ValueError: If input shapes are inconsistent or labels out of range.
    """
    if njobs not in (None, 0, 1):
        warnings.warn(
            "njobs is ignored in the pure-Python implementation.", RuntimeWarning
        )

    X, y, centers, valid_centers, valid_cols = _prepare_min_mistakes_inputs(
        X, y, centers, valid_centers, valid_cols
    )

    n_samples, n_features = X.shape
    n_centers = centers.shape[0]

    if n_samples == 0 or n_features == 0 or n_centers == 0:
        return None

    # Count points per center label
    centers_count: NDArray[np.int64] = np.bincount(y, minlength=n_centers).astype(
        np.int64
    )

    best_col = -1
    best_threshold: Optional[float] = None
    min_mistakes = np.iinfo(np.int64).max

    for col in range(n_features):
        if valid_cols[col] != 1:
            continue
        result = _best_threshold_min_mistakes_for_column(
            X=X,
            y=y,
            centers=centers,
            valid_centers=valid_centers,
            centers_count=centers_count,
            col=col,
        )
        if result is None:
            continue
        threshold, mistakes = result
        if mistakes < min_mistakes:
            min_mistakes = mistakes
            best_col = col
            best_threshold = threshold

    if best_col == -1 or best_threshold is None:
        return None
    return IMM_Cut(col=int(best_col), threshold=float(best_threshold))


def _prepare_min_mistakes_inputs(
    X: NDArray[np.floating],
    y: NDArray[np.integer],
    centers: NDArray[np.floating],
    valid_centers: NDArray[np.integer],
    valid_cols: NDArray[np.integer],
) -> tuple[
    NDArray[np.floating],
    NDArray[np.int64],
    NDArray[np.floating],
    NDArray[np.int8],
    NDArray[np.int8],
]:
    """Validate and normalize inputs for the min mistakes computation."""
    X = np.asarray(X, dtype=float, order="C")
    y = np.asarray(y, dtype=np.int64)
    centers = np.asarray(centers, dtype=float, order="C")
    valid_centers = np.asarray(valid_centers, dtype=np.int8)
    valid_cols = np.asarray(valid_cols, dtype=np.int8)

    n_samples, n_features = X.shape
    n_centers, n_features_c = centers.shape

    if n_features != n_features_c:
        raise ValueError(
            "X and centers must have the same number of columns (features)."
        )
    if y.shape != (n_samples,):
        raise ValueError(
            "y must be a 1D array with length equal to the number of samples in X."
        )
    if valid_centers.shape != (n_centers,):
        raise ValueError(
            "valid_centers must be a 1D array with length equal to the number of centers."
        )
    if valid_cols.shape != (n_features,):
        raise ValueError(
            "valid_cols must be a 1D array with length equal to the number of features."
        )

    if n_centers > 0:
        if y.min(initial=0) < 0 or y.max(initial=-1) >= n_centers:
            raise ValueError("y contains center indices out of range [0, n_centers).")

    return X, y, centers, valid_centers, valid_cols


def _best_threshold_min_mistakes_for_column(
    X: NDArray[np.floating],
    y: NDArray[np.int64],
    centers: NDArray[np.floating],
    valid_centers: NDArray[np.int8],
    centers_count: NDArray[np.int64],
    col: int,
) -> Optional[Tuple[float, int]]:
    """
    Compute the best threshold for a single column that minimizes mistakes.

    Returns:
        (best_threshold, min_mistakes) or None if no valid threshold found.
    """
    n_samples = X.shape[0]
    n_centers = centers.shape[0]
    if n_samples < 2:
        return None

    # Stable sort to handle equal values predictably.
    data_order = np.argsort(X[:, col], kind="mergesort")
    centers_order = np.argsort(centers[:, col], kind="mergesort")

    # Early exit if no valid center
    if not np.any(valid_centers == 1):
        return None

    # Any valid threshold must be < maximum valid center value.
    max_valid_center_val = float(np.max(centers[valid_centers == 1, col]))

    # Sweep state
    left_centers_count = np.zeros(n_centers, dtype=np.int64)
    ix = 0  # index over data_order
    ic = 0  # index over centers_order

    # Advance to first valid center
    while ic < n_centers and valid_centers[centers_order[ic]] == 0:
        ic += 1
    if ic >= n_centers:
        return None

    threshold = float(centers[centers_order[ic], col])
    is_center_threshold = True
    mistakes = 0

    # Accumulate points <= initial threshold
    while ix < n_samples and X[data_order[ix], col] <= threshold:
        cidx = int(y[data_order[ix]])
        left_centers_count[cidx] += 1
        if centers[cidx, col] >= threshold:
            mistakes += 1
        ix += 1

    best_threshold: Optional[float] = None
    min_mistakes = np.iinfo(np.int64).max
    valid_found = False

    # Corner case: exactly one point on the right
    if ix == n_samples - 1:
        cc_mistakes = 0
        for c in range(n_centers):
            if valid_centers[c] and centers[c, col] > threshold:
                cc_mistakes += int(centers_count[c])
        last_point_center = int(y[data_order[n_samples - 1]])
        if centers[last_point_center, col] > threshold:
            cc_mistakes -= 1
        if cc_mistakes < min_mistakes:
            min_mistakes = cc_mistakes
            best_threshold = threshold
            valid_found = True

    # Main sweep
    while ix < n_samples - 1 and ic < n_centers:
        if threshold >= max_valid_center_val:
            break

        if not is_center_threshold:
            # Move one data point to the left partition
            cidx = int(y[data_order[ix]])
            left_centers_count[cidx] += 1
            if centers[cidx, col] >= threshold:
                mistakes += 1
            else:
                mistakes -= 1
            ix += 1
        else:
            # Cross a center: update mistakes by flipping its points
            center_idx = int(centers_order[ic])
            mistakes += int(centers_count[center_idx]) - 2 * int(
                left_centers_count[center_idx]
            )
            ic += 1
            while ic < n_centers and valid_centers[centers_order[ic]] == 0:
                ic += 1

        prev_threshold = threshold

        # Determine next threshold candidate
        next_point_val = float(X[data_order[ix], col]) if ix < n_samples else INF
        next_center_val = (
            float(centers[centers_order[ic], col]) if ic < n_centers else INF
        )

        if next_point_val <= next_center_val:
            threshold = next_point_val
            is_center_threshold = False
        else:
            threshold = next_center_val
            is_center_threshold = True

        # Record the best distinct threshold so far
        if prev_threshold != threshold and mistakes < min_mistakes:
            min_mistakes = mistakes
            best_threshold = prev_threshold
            valid_found = True

    if not valid_found or best_threshold is None:
        return None
    return float(best_threshold), int(min_mistakes)


def get_min_surrogate_cut(
    X: NDArray[np.floating],
    X_center_dot: NDArray[np.floating],
    X_sum_all_center_dot: NDArray[np.floating],
    centers_norm_sqr: NDArray[np.floating],
    njobs: Optional[int] = None,
) -> Optional[Surrogate_Cut]:
    """
    Find the feature and threshold that minimize the surrogate (sum-of-costs) objective.

    For a given threshold on a column, the left and right partitions independently
    choose their best center to minimize:
        n_left * ||c||^2 - 2 * sum_{i in left} <x_i, c>
        n_right * ||c||^2 - 2 * sum_{i in right} <x_i, c>

    Args:
        X: Array of shape (n_samples, n_features).
        X_center_dot: Array of shape (n_samples, n_centers) where entry (i, j)
            is <x_i, c_j>.
        X_sum_all_center_dot: Array of shape (n_centers,), sum over samples of <x_i, c_j>.
        centers_norm_sqr: Array of shape (n_centers,), squared L2 norms of centers.
        njobs: Ignored. Present for API compatibility.

    Returns:
        Surrogate_Cut if a valid cut is found, else None.

    Raises:
        ValueError: If input shapes are inconsistent.
    """
    if njobs not in (None, 0, 1):
        warnings.warn(
            "njobs is ignored in the pure-Python implementation.", RuntimeWarning
        )

    X, X_center_dot, X_sum_all_center_dot, centers_norm_sqr = _prepare_surrogate_inputs(
        X, X_center_dot, X_sum_all_center_dot, centers_norm_sqr
    )

    n_samples, n_features = X.shape
    if n_samples < 2 or n_features == 0:
        return None

    best_col = -1
    best_threshold: Optional[float] = None
    best_cost = INF
    best_left_center = -1
    best_right_center = -1

    for col in range(n_features):
        result = _best_surrogate_for_column(
            X=X,
            X_center_dot=X_center_dot,
            centers_norm_sqr=centers_norm_sqr,
            X_sum_all_center_dot=X_sum_all_center_dot,
            col=col,
        )
        if result is None:
            continue
        threshold, cost, left_center, right_center = result
        if cost < best_cost:
            best_cost = cost
            best_col = col
            best_threshold = threshold
            best_left_center = left_center
            best_right_center = right_center

    if best_col == -1 or best_threshold is None:
        return None

    return Surrogate_Cut(
        col=int(best_col),
        threshold=float(best_threshold),
        cost=float(best_cost),
        center_left=int(best_left_center),
        center_right=int(best_right_center),
    )


def _prepare_surrogate_inputs(
    X: NDArray[np.floating],
    X_center_dot: NDArray[np.floating],
    X_sum_all_center_dot: NDArray[np.floating],
    centers_norm_sqr: NDArray[np.floating],
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """Validate and normalize inputs for the surrogate objective computation."""
    X = np.asarray(X, dtype=float, order="C")
    X_center_dot = np.asarray(X_center_dot, dtype=float, order="C")
    X_sum_all_center_dot = np.asarray(X_sum_all_center_dot, dtype=float)
    centers_norm_sqr = np.asarray(centers_norm_sqr, dtype=float)

    n_samples = X.shape[0]
    if X_center_dot.shape[0] != n_samples:
        raise ValueError(
            "X_center_dot must have the same number of rows as X (n_samples)."
        )
    n_centers = X_center_dot.shape[1]
    if X_sum_all_center_dot.shape != (n_centers,):
        raise ValueError("X_sum_all_center_dot must have shape (n_centers,).")
    if centers_norm_sqr.shape != (n_centers,):
        raise ValueError("centers_norm_sqr must have shape (n_centers).")

    return X, X_center_dot, X_sum_all_center_dot, centers_norm_sqr


def _best_surrogate_for_column(
    X: NDArray[np.floating],
    X_center_dot: NDArray[np.floating],
    centers_norm_sqr: NDArray[np.floating],
    X_sum_all_center_dot: NDArray[np.floating],
    col: int,
) -> Optional[Tuple[float, float, int, int]]:
    """
    Compute the best surrogate cut for a single column.

    Returns:
        (best_threshold, best_total_cost, best_center_left, best_center_right) or None
        if no valid threshold exists.
    """
    n_samples = X.shape[0]
    if n_samples < 2:
        return None

    # Stable sort to ensure predictable handling of ties.
    data_order = np.argsort(X[:, col], kind="mergesort")

    # Initialize with the first point on the left
    ix = 0
    threshold = float(X[data_order[0], col])
    xcd_left = X_center_dot[data_order[0]].astype(float, copy=False)
    xcd_right = X_sum_all_center_dot.astype(float, copy=True) - xcd_left
    n_left = 1
    n_right = n_samples - 1

    best_threshold: Optional[float] = None
    best_cost = INF
    best_left_center = -1
    best_right_center = -1
    found = False

    while ix < n_samples - 1:
        ix += 1
        prev_threshold = threshold
        threshold = float(X[data_order[ix], col])

        if prev_threshold != threshold:
            # Evaluate best centers for left and right partitions
            left_costs = n_left * centers_norm_sqr - 2.0 * xcd_left
            right_costs = n_right * centers_norm_sqr - 2.0 * xcd_right

            left_center = int(np.argmin(left_costs))
            right_center = int(np.argmin(right_costs))
            total_cost = float(left_costs[left_center] + right_costs[right_center])
        else:
            # No evaluation on identical thresholds (only update accumulators)
            total_cost = INF
            left_center = -1
            right_center = -1

        # Move current point from right to left
        xcd_curr = X_center_dot[data_order[ix]]
        xcd_left = xcd_left + xcd_curr
        xcd_right = xcd_right - xcd_curr
        n_left += 1
        n_right -= 1

        if prev_threshold != threshold and total_cost < best_cost:
            best_cost = total_cost
            best_threshold = prev_threshold
            best_left_center = left_center
            best_right_center = right_center
            found = True

    if not found or best_threshold is None:
        return None
    return (
        float(best_threshold),
        float(best_cost),
        int(best_left_center),
        int(best_right_center),
    )
