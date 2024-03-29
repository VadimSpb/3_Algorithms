import math
import numpy as np
import pandas as pd
from numpy import ndarray

"""
This code was taken from  of Daniel Korenblum's post. Sources:
https://towardsdatascience.com/random-forests-and-decision-trees-from-scratch-in-python-3e4fa5ae4249
https://gist.github.com/TimeTraveller-San/9667d80e97fe5a6fbd25f8f8e7539797#file-randomforest_complete-py 
"""


class DecisionTree:
    def __init__(self, x, y, n_features, f_idxs, idxs, depth=10, min_leaf=5):
        self.x, self.y, self.idxs, self.min_leaf, self.f_idxs = x, y, idxs, min_leaf, f_idxs
        self.depth = depth
        # print(f_idxs)
        # print(self.depth)
        self.n_features = n_features
        self.n, self.c = len(idxs), x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.find_varsplit()

    @staticmethod
    def std_agg(cnt, s1, s2):
        return math.sqrt((s2 / cnt) - (s1 / cnt) ** 2)

    def find_varsplit(self):
        for i in self.f_idxs:
            self.find_better_split(i)
        if self.is_leaf:
            return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        lf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        rf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        self.lhs = DecisionTree(self.x, self.y, self.n_features, lf_idxs, self.idxs[lhs], depth=self.depth - 1,
                                min_leaf=self.min_leaf)
        self.rhs = DecisionTree(self.x, self.y, self.n_features, rf_idxs, self.idxs[rhs], depth=self.depth - 1,
                                min_leaf=self.min_leaf)

    def find_better_split(self, var_idx):
        x, y = self.x.values[self.idxs, var_idx], self.y[self.idxs]
        sort_idx = np.argsort(x)
        sort_y, sort_x = y[sort_idx], x[sort_idx]
        rhs_cnt, rhs_sum, rhs_sum2 = self.n, sort_y.sum(), (sort_y ** 2).sum()
        lhs_cnt, lhs_sum, lhs_sum2 = 0, 0., 0.

        for i in range(0, self.n - self.min_leaf - 1):
            xi, yi = sort_x[i], sort_y[i]
            lhs_cnt += 1
            rhs_cnt -= 1
            lhs_sum += yi
            rhs_sum -= yi
            lhs_sum2 += yi ** 2
            rhs_sum2 -= yi ** 2
            if i < self.min_leaf or xi == sort_x[i + 1]:
                continue

            lhs_std = self.std_agg(lhs_cnt, lhs_sum, lhs_sum2)
            rhs_std = self.std_agg(rhs_cnt, rhs_sum, rhs_sum2)
            curr_score = lhs_std * lhs_cnt + rhs_std * rhs_cnt
            if curr_score < self.score:
                self.var_idx, self.score, self.split = var_idx, curr_score, xi

    @property
    def split_name(self):
        return self.x.columns[self.var_idx]

    @property
    def split_col(self):
        return self.x.values[self.idxs, self.var_idx]

    @property
    def is_leaf(self):
        return self.score == float('inf') or self.depth <= 0

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf:
            return self.val
        t = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return t.predict_row(xi)


class RandomForestRegressor:
    """
    A random forest regressor.
    x : independent variables of training set. To keep things minimal and simple I am not creating a separate fit method hence the base class constructor will accept the training set.
    y : the corresponding dependent variables necessary for supervised learning (Random forest is a supervised learning technique)
    n_trees : number of uncorrelated trees we ensemble to create the random forest.
    n_features : the number of features to sample and pass onto each tree, this is where feature bagging happens. It can either be sqrt, log2 or an integer. In case of sqrt, the number of features sampled to each tree is square root of total features and log base 2 of total features in case of log2.
    sample_sz : the number of rows randomly selected and passed onto each tree. This is usually equal to total number of rows but can be reduced to increase performance and decrease correlation of trees in some cases (bagging of trees is a completely separate machine learning technique)
    depth : depth of each decision tree. Higher depth means more number of splits which increases the over fitting tendency of each tree but since we are aggregating several uncorrelated trees, over fitting of individual trees hardly bothers the whole forest.
    min_leaf : minimum number of rows required in a node to cause further split. Lower the min_leaf, higher the depth of the tree.
    """
    def __init__(self,
                 x,
                 y,
                 n_trees,
                 n_features=None,
                 sample_sz=None,
                 depth=10,
                 min_leaf=5,
                 random_seed=42):
        if type(x) == 'pandas.core.frame.DataFrame':
            self.x = x
        else:
            self.x = pd.DataFrame(x)
        self.y = y
        if n_features == 'sqrt':
            self.n_features = int(np.sqrt(x.shape[1]))
        elif n_features == 'log2':
            self.n_features = int(np.log2(x.shape[1]))
        elif n_features is None:
            self.n_features = x.shape[1]
        else:
            self.n_features = n_features
        # print(self.n_features, "sha: ", x.shape[1])
        if sample_sz is None:
            self.sample_sz = x.shape[0]
        else:
            self.sample_sz = sample_sz
        self.depth = depth
        self.min_leaf = min_leaf
        np.random.seed(random_seed)
        self.trees = [self.create_tree() for i in range(n_trees)]

    def create_tree(self):
        idxs = np.random.permutation(len(self.y))[:self.sample_sz]
        f_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        return DecisionTree(self.x.iloc[idxs], self.y[idxs], self.n_features, f_idxs,
                            idxs=np.array(range(self.sample_sz)), depth=self.depth, min_leaf=self.min_leaf)

    def predict(self, x):
        return np.mean([t.predict(x) for t in self.trees], axis=0)


class RandomForestClassifier(RandomForestRegressor):

    def __init__(self, tolerance=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tolerance = tolerance

    def predict(self, x):
        y_pred: ndarray = np.mean([t.predict(x) for t in self.trees], axis=0)
        return [np.ceil(i) if i > self.tolerance else np.floor(i) for i in y_pred]

