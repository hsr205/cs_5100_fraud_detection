from collections import Counter

import numpy as np


class DecisionTree:
    """
    A simple implementation of a Decision Tree for binary classification tasks.
    """

    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Initializes the DecisionTree object.

        Parameters:
        - max_depth: The maximum depth of the tree. If None, the tree grows until pure or min_samples_split is reached.
        - min_samples_split: The minimum number of samples required to split an internal node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        """
        Fits the Decision Tree to the data using a recursive approach.

        Parameters:
        - X: Feature matrix (numpy array of shape (n_samples, n_features)).
        - y: Target labels (numpy array of shape (n_samples,)).
        - depth: Current depth of the tree.

        Returns:
        - The tree structure as a tuple (feature_index, threshold, left_subtree, right_subtree).
        """
        num_samples, num_features = X.shape
        # Stopping conditions
        if num_samples >= self.min_samples_split and depth < self.max_depth:
            best_feature, best_threshold = self._best_split(X, y)
            if best_feature is not None:
                # Split the data based on the best split
                left_indices, right_indices = self._split(X[:, best_feature], best_threshold)
                # Recursively build the left and right subtrees
                left_subtree = self.fit(X[left_indices, :], y[left_indices], depth + 1)
                right_subtree = self.fit(X[right_indices, :], y[right_indices], depth + 1)
                return (best_feature, best_threshold, left_subtree, right_subtree)
        # Return the most common label in the current node
        return Counter(y).most_common(1)[0][0]

    def _best_split(self, X, y):
        """
        Finds the best split for the data by minimizing the Gini impurity.

        Parameters:
        - X: Feature matrix.
        - y: Target labels.

        Returns:
        - best_feature: Index of the best feature for the split.
        - best_threshold: Best threshold value for the split.
        """
        num_samples, num_features = X.shape
        best_gini = 1.0
        best_feature, best_threshold = None, None
        unique_classes = set(y)

        for feature_index in range(num_features):
            thresholds, classes = zip(*sorted(zip(X[:, feature_index], y)))
            num_left = Counter()
            num_right = Counter(classes)

            for i in range(1, num_samples):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1

                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in unique_classes)
                gini_right = 1.0 - sum((num_right[x] / (num_samples - i)) ** 2 for x in unique_classes)
                gini = (i * gini_left + (num_samples - i) * gini_right) / num_samples

                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2
        return best_feature, best_threshold

    def _split(self, X_column, split_threshold):
        """
        Splits the data based on the given threshold.

        Parameters:
        - X_column: The column of data to split.
        - split_threshold: The threshold value for splitting.

        Returns:
        - left_indices: Indices of the left split.
        - right_indices: Indices of the right split.
        """
        left_indices = np.argwhere(X_column < split_threshold).flatten()
        right_indices = np.argwhere(X_column >= split_threshold).flatten()
        return left_indices, right_indices

    def predict_sample(self, x, tree):
        """
        Predicts the label for a single sample.

        Parameters:
        - x: A single sample.
        - tree: The tree structure.

        Returns:
        - The predicted label.
        """
        if not isinstance(tree, tuple):
            return tree
        feature, threshold, left_subtree, right_subtree = tree
        if x[feature] < threshold:
            return self.predict_sample(x, left_subtree)
        else:
            return self.predict_sample(x, right_subtree)

    def predict(self, X):
        """
        Predicts the labels for a set of samples.

        Parameters:
        - X: Feature matrix (n_samples, n_features).

        Returns:
        - Array of predictions.
        """
        return np.array([self.predict_sample(x, self.tree) for x in X])


class RandomForest:
    """
    A simple implementation of a Random Forest classifier.
    """

    def __init__(self, num_trees=7, max_depth=9, min_samples_split=7):
        """
        Initializes the RandomForest object.

        Parameters:
        - num_trees: Number of decision trees in the forest.
        - max_depth: Maximum depth of each decision tree.
        - min_samples_split: Minimum samples required to split a node.
        """
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def _bootstrap_sample(self, X, y):
        """
        Creates a bootstrap sample from the dataset.

        Parameters:
        - X: Feature matrix.
        - y: Target labels.

        Returns:
        - A tuple (X_sample, y_sample) of the sampled data.
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        """
        Trains the RandomForest on the given data.

        Parameters:
        - X: Feature matrix.
        - y: Target labels.
        """
        self.trees = []
        for _ in range(self.num_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.tree = tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predicts the labels for the given data.

        Parameters:
        - X: Feature matrix.

        Returns:
        - Array of predictions.
        """
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_predictions)
