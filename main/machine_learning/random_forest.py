from typing import Any
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker

import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if num_samples >= self.min_samples_split and depth < self.max_depth:
            best_feature, best_threshold = self._best_split(X, y)
            if best_feature is not None:
                left_indices, right_indices = self._split(X[:, best_feature], best_threshold)
                left_subtree = self.fit(X[left_indices, :], y[left_indices], depth + 1)
                right_subtree = self.fit(X[right_indices, :], y[right_indices], depth + 1)
                return (best_feature, best_threshold, left_subtree, right_subtree)
        return Counter(y).most_common(1)[0][0]

    def _best_split(self, X, y):
      num_samples, num_features = X.shape
      best_gini = 1.0
      best_feature, best_threshold = None, None
      unique_classes = set(y)  # Get the unique classes in y

      for feature_index in range(num_features):
        thresholds, classes = zip(*sorted(zip(X[:, feature_index], y)))
        num_left = Counter()  # Initialize as Counter to handle arbitrary labels
        num_right = Counter(
          classes)  # Start with all instances in the right split

        for i in range(1, num_samples):
          c = classes[i - 1]
          num_left[c] += 1
          num_right[c] -= 1
          gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in unique_classes)
          gini_right = 1.0 - sum(
              (num_right[x] / (num_samples - i)) ** 2 for x in unique_classes)
          gini = (i * gini_left + (num_samples - i) * gini_right) / num_samples

          if thresholds[i] == thresholds[i - 1]:
            continue
          if gini < best_gini:
            best_gini = gini
            best_feature = feature_index
            best_threshold = (thresholds[i] + thresholds[i - 1]) / 2
      return best_feature, best_threshold

    def _split(self, X_column, split_threshold):
        left_indices = np.argwhere(X_column < split_threshold).flatten()
        right_indices = np.argwhere(X_column >= split_threshold).flatten()
        return left_indices, right_indices

    def predict_sample(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feature, threshold, left_subtree, right_subtree = tree
        if x[feature] < threshold:
            return self.predict_sample(x, left_subtree)
        else:
            return self.predict_sample(x, right_subtree)

    def predict(self, X):
        return np.array([self.predict_sample(x, self.tree) for x in X])


class RandomForest:
  def __init__(self, num_trees=3, max_depth=3, min_samples_split=2):
    self.num_trees = num_trees
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.trees = []

  def _bootstrap_sample(self, X, y):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, n_samples, replace=True)
    return X[indices], y[indices]

  def fit(self, X, y):
    self.trees = []
    for _ in range(self.num_trees):
      tree = DecisionTree(max_depth=self.max_depth,
                          min_samples_split=self.min_samples_split)
      X_sample, y_sample = self._bootstrap_sample(X, y)
      tree.tree = tree.fit(X_sample, y_sample)
      self.trees.append(tree)

  def predict(self, X):
    tree_predictions = np.array([tree.predict(X) for tree in self.trees])
    # Majority voting
    return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0],
                               axis=0, arr=tree_predictions)
