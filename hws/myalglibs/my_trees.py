import numpy as np
import random
import math


class Node:
    """
    Base class for tree's nodes.

    Warning: This class should not be used directly.
    Use derived classes instead.

    index : индекс признака, по которому ведется сравнение с порогом в этом узле
    t : значение порога
    true_branch : поддерево, удовлетворяющее условию в узле
    false_branch : поддерево, не удовлетворяющее условию в узле
    max_leaf : максимальное количество листьев в узле
    leafs : количество листьев в узле

    """

    def __init__(self, index=None, t=None, true_branch=None, false_branch=None):
        self.index = index
        self.t = t
        self.true_branch = true_branch
        self.false_branch = false_branch
#        self.max_leaf = 5
        self.leafs = self.count_leafs()

    def count_leafs(self):
        branches = [self.true_branch, self.false_branch]
        leafs = 0
        for branch in branches:
            if type(branch) is Leaf:
                leafs += 1
            elif type(branch) is Node:
                leafs += branch.leafs
        return leafs


class Leaf:
    """
    Base class for tree's nodes.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.prediction = self.predict()

    def predict(self):
        # подсчет количества объектов разных классов
        classes = {}  # сформируем словарь "класс: количество объектов"
        for label in self.labels:
            if label not in classes:
                classes[label] = 0
            classes[label] += 1
        #  найдем класс, количество объектов которого будет максимальным в этом листе и вернем его
        prediction = max(classes, key=classes.get)
        return prediction


class OldLessonTree:
    def __init__(self):
        self.node = Node()
        pass

    @staticmethod
    def _gini(labels):
        classes = {}
        for label in labels:
            if label not in classes:
                classes[label] = 0
            classes[label] += 1
        impurity = 1
        for label in classes:
            p = classes[label] / len(labels)
            impurity -= p ** 2
        return impurity

    @staticmethod
    def _get_bootstrap(data, labels, N):
        n_samples = data.shape[0]
        bootstrap = []
        for i in range(N):
            b_data = np.zeros(data.shape)
            b_labels = np.zeros(labels.shape)

            for j in range(n_samples):
                sample_index = random.randint(0, n_samples - 1)
                b_data[j] = data[sample_index]
                b_labels[j] = labels[sample_index]
            bootstrap.append((b_data, b_labels))
        return bootstrap

    @staticmethod
    def _get_subsample(len_sample):
        # будем сохранять не сами признаки, а их индексы
        sample_indexes = [i for i in range(len_sample)]

        len_subsample = int(np.sqrt(len_sample))
        subsample = []

        random.shuffle(sample_indexes)
        for _ in range(len_subsample):
            subsample.append(sample_indexes.pop())

        return subsample

    def _quality(self, left_labels, right_labels, current_gini):
        p = float(left_labels.shape[0]) / (left_labels.shape[0] + right_labels.shape[0])
        return current_gini - p * self._gini(left_labels) - (1 - p) * self._gini(right_labels)

    @staticmethod
    def _split(data, labels, index, t):
        """Разбиение датасета в узле"""
        left = np.where(data[:, index] <= t)
        right = np.where(data[:, index] > t)

        true_data = data[left]
        false_data = data[right]
        true_labels = labels[left]
        false_labels = labels[right]

        return true_data, false_data, true_labels, false_labels

    def _find_best_split(self, data, labels):
        """Нахождение наилучшего разбиения"""
        min_leaf = 1

        current_gini = self._gini(labels)

        best_quality = 0
        best_t = None
        best_index = None

        n_features = data.shape[1]

        # выбор индекса из подвыборки длиной sqrt(n_features)
        subsample = self._get_subsample(n_features)

        for index in subsample:
            t_values = [row[index] for row in data]

            for t in t_values:
                true_data, false_data, true_labels, false_labels = self._split(data, labels, index, t)
                #  пропускаем разбиения, в которых в узле остается менее 5 объектов
                if len(true_data) < min_leaf or len(false_data) < min_leaf:
                    continue

                current_quality = self._quality(true_labels, false_labels, current_gini)

                #  выбираем порог, на котором получается максимальный прирост качества
                if current_quality > best_quality:
                    best_quality, best_t, best_index = current_quality, t, index

        return best_quality, best_t, best_index

    def build_tree(self, data, labels):

        quality, t, index = self._find_best_split(data, labels)

        #  Базовый случай - прекращаем рекурсию, когда нет прироста в качества
        if quality == 0:
            return Leaf(data, labels)

        true_data, false_data, true_labels, false_labels = self._split(data, labels, index, t)

        # Рекурсивно строим два поддерева
        true_branch = self.build_tree(true_data, true_labels)
        false_branch = self.build_tree(false_data, false_labels)

        # Возвращаем класс узла со всеми поддеревьями, то есть целого дерева
        self.node = Node(index, t, true_branch, false_branch)
        return Node(index, t, true_branch, false_branch)

    def _classify_object(self, obj, node):
        if isinstance(node, Leaf):
            answer = node.prediction
            return answer
        if obj[node.index] <= node.t:
            return self._classify_object(obj, node.true_branch)
        else:
            return self._classify_object(obj, node.false_branch)

    def predict(self, data, tree):
        classes = []
        for obj in data:
            prediction = self._classify_object(obj, tree)
            classes.append(prediction)
        return classes

    def print_tree(self, node, spacing=""):

        # Если лист, то выводим его прогноз
        if isinstance(node, Leaf):
            print(spacing + "Прогноз:", node.prediction)
            return

        # Выведем значение индекса и порога на этом узле
        print(spacing + 'Индекс', str(node.index))
        print(spacing + 'Порог', str(node.t))

        # Рекурсионный вызов функции на положительном поддереве
        print(spacing + '--> True:')
        self.print_tree(node.true_branch, spacing + "  ")

        # Рекурсионный вызов функции на положительном поддереве
        print(spacing + '--> False:')
        self.print_tree(node.false_branch, spacing + "  ")

class MyRandomForestClassifier:

    def __init__(self, n_trees=None):
        self.n_trees = None

    


# class MyBaseTree:
#     def __init__(self, criterion='gini'):
#         self.data = None
#         self.labels = None
#         self.n_feature = None
#         self.min_leaf = None
#         self.criterion = criterion
#         pass
#
#     @staticmethod
#     def _gini(labels):
#         """
#         calculating Gini coefficient
#         it is O(n) in time and memory, where n = len(x)
#         learn more:
#         https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python/48999797#48999797
#         https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy
#         """
#         classes = {}
#         for label in labels:
#             if label not in classes:
#                 classes[label] = 0
#             classes[label] += 1
#             #  расчет критерия
#         impurity = 1
#         for label in classes:
#             p = classes[label] / len(labels)
#             impurity -= p ** 2
#         return impurity
#
#     @staticmethod
#     def _entropy(labels):
#         """
#         Computing entropy.
#         Learn more:
#         https://gist.github.com/jaradc/eeddf20932c0347928d0da5a09298147
#         """
#         value, counts = np.unique(labels, return_counts=True)
#         norm_counts = counts / counts.sum()
#         return -(norm_counts * np.log(norm_counts) / np.log(np.exp(1))).sum()
#
#     def _get_criteria(self, labels):
#         """
#         The function to measure the quality of a split.
#         Supported criteria are “gini” for the Gini impurity
#         and “entropy” for the information gain.
#         Criterion: {“gini”, “entropy”}, default = "jini"
#         """
#         if self.criterion == 'gini':
#             return self._gini(labels)
#         elif self.criterion == 'entropy':
#             return self._entropy(labels)
#         else:
#             raise ValueError("Parameters for criteries can be either 'gini' or 'entropy'")
#
#     @staticmethod
#     def _node_split(data, labels, index, t):
#         left = np.where(data[:, index] <= t)
#         right = np.where(data[:, index] > t)
#         true_data = data[left]
#         false_data = data[right]
#         true_labels = labels[left]
#         false_labels = labels[right]
#         return true_data, false_data, true_labels, false_labels
#
#     def _quality(self, left_labels, right_labels, current_criteria):
#         """
#         доля выборки, ушедшая в левое поддерево
#         """
#         p = float(left_labels.shape[0]) / (left_labels.shape[0] + right_labels.shape[0])
#         return current_criteria - p * self._get_criteria(left_labels) - (1 - p) * self._get_criteria(right_labels)
#
#     def _find_best_split(self):
#
#         current_criteria = self._get_criteria(self.labels)
#         best_quality = 0
#         best_t = None
#         best_index = None
#
#         for index in range(self.n_features):
#             t_values = [row[index] for row in self.data]
#
#             for t in t_values:
#                 true_data, false_data, true_labels, false_labels = self._node_split(self.data, self.labels, index, t)
#                 #  пропускаем разбиения, в которых в узле остается менее 5 объектов
#                 if len(true_data) < self.min_leaf or len(false_data) < self.min_leaf:
#                     continue
#
#                 current_quality = self._quality(true_labels, false_labels, current_criteria)
#
#                 #  выбираем порог, на котором получается максимальный прирост качества
#                 if current_quality > best_quality:
#                     best_quality, best_t, best_index = current_quality, t, index
#
#         return best_quality, best_t, best_index
#
#     def _build_tree(self, data=None, labels=None, max_leaf=None):
#         if data is None:
#             data = self.data
#         if labels is None:
#             labels = self.labels
#         if max_leaf is None:
#             max_leaf = self.max_leaf
#         quality, t, index = self._find_best_split()
#         if quality == 0 or self.max_leaf == 1:
#             return Leaf(self.data, self.labels)
#         node = Node(index, t, None, None)
#         node.max_leaf = max_leaf
#         true_data, false_data, true_labels, false_labels = self._node_split(data, labels, index, t)
#         true_branch = self._build_tree(data=true_data, labels=true_labels, max_leaf=node.max_leaf - 1)
#         false_branch = self._build_tree(data=false_data, labels=false_labels, max_leaf=node.max_leaf - node.leafs)
#         return Node(index, t, true_branch, false_branch)
#
#     def fit(self, data, labels, min_leaf=2, max_leaf=5, n_features=None, criterion='gini'):
#         self.data = data
#         self.labels = labels
#         self.min_leaf = min_leaf
#         self.max_leaf = max_leaf
#         self.criterion = criterion
#         if n_features is None:
#             self.n_features = self.data.shape[1]
#         else:
#             self.n_features = n_features
#         return self._build_tree()
#
#     def _classify_object(self, obj, node) -> object:
#         if isinstance(node, Leaf):
#             answer = node.prediction
#             return answer
#         if obj[node.index] <= node.t:
#             return self._classify_object(obj, node.true_branch)
#         else:
#             return self._classify_object(obj, node.false_branch)
#
#     def predict(self, data, tree):
#
#         classes = []
#         for obj in data:
#             prediction = self._classify_object(obj, tree)
#             classes.append(prediction)
#         return classes

