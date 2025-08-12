import numpy as np
class MyDecisionTree:
    def __init__(self, max_depth=5, min_samples=5):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None

    class Node:
        def __init__(self, mean, split_feature=None, split_value=None):
            self.mean = mean
            self.split_feature = split_feature
            self.split_value = split_value
            self.left = None
            self.right = None

    def _gini(self, y_left, y_right):
        p_left = y_left.mean()
        p_right = y_right.mean()
        return 2*p_left*(1-p_left) + 2*p_right*(1-p_right)

    def _best_split(self, X, y):
        best = (None, None, np.inf)
        
        for p in range(X.shape[1]):
            values = np.unique(X[:, p])
            
            for v in values:
                left_mask = X[:, p] < v
                if left_mask.sum() < self.min_samples or (~left_mask).sum() < self.min_samples:
                    continue
                
                gini = self._gini(y[left_mask], y[~left_mask])
                imbalance = abs(left_mask.sum() - len(X)/2) / len(X)
                score = gini + 0.5*imbalance
                
                if score < best[2]:
                    best = (p, v, score)
        
        return best

    def _build_tree(self, X, y, depth):
        node = self.Node(y.mean())
        
        if depth == 0 or len(y) < 2*self.min_samples:
            return node
            
        p, v, _ = self._best_split(X, y)
        if p is None:
            return node
            
        left_mask = X[:, p] < v
        node.split_feature = p
        node.split_value = v
        node.left = self._build_tree(X[left_mask], y[left_mask], depth-1)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], depth-1)
        
        return node

    def fit(self, X, y):
        self.root = self._build_tree(X, y, self.max_depth)

    def _predict(self, x, node):
        if node.split_feature is None:
            return node.mean
        if x[node.split_feature] < node.split_value:
            return self._predict(x, node.left)
        return self._predict(x, node.right)

    def predict(self, X):
        return np.array([self._predict(x, self.root) for x in X])