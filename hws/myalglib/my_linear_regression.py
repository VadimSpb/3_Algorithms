import numpy as np
# import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_regression


class MyLinearRegression:

    def __init__(self, eta=0.9, max_iter=1e4, min_weight_dist=1e-8):
        """
        :param eta: шаг градиентного спуска
        :param max_iter: ограничение максимального числа итераций
        :param min_weight_dist: критерий сходимости (разница весов, при которой алгоритм останавливается)
        """
        self.eta = eta
        self.max_iter = max_iter
        self.min_weight_dist = min_weight_dist

    def _mserror(self, X, y_real):
        """
        :param X: известные атрибуты (фичи) объектов, по которым мы собираемся получить предсказание
        :param y_real: известный целевой атрибут, с которым сравнивается предсказанный результат
        :return: возвращает среднюю квадратичную ошибку
        """
        y = X.dot(self.w.T) + self.w0
        return np.sum((y - y_real)**2) / y_real.shape[0]

    def _mserror_grad(self, X, y_real):
        """
        расчёт градиента ошибки
        :param X: известные атрибуты (фичи) объектов, по которым мы собираемся получить предсказание
        :param y_real: известный целевой атрибут, с которым сравнивается предсказанный результат
        :return: возвращает градиент ошибки
        """
        delta = (X.dot(self.w.T) + self.w0 - y_real)
        return 2 * delta.T.dot(X) / y_real.shape[0], np.sum(2 * delta) / y_real.shape[0]

    def _optimize(self, X, Y):
        """
        Оптимизация коэффициентов (поиск оптимальных коэффициентов w и w0 для уравнения линейной регресии)
        :param X: известные атрибуты (фичи) объектов, по которым мы собираемся получить предсказание
        :param Y: известный целевой атрибут
        :return: оптимизированные веса линейной регрессии
        """
        iter_num = 0
        weight_dist = np.inf
        self.w = np.zeros((1, X.shape[1]))
        self.w0 = 0
        w_ls,  w0_ls = [], []
        w_ls.append(self.w)
        w0_ls.append(self.w0)

        while weight_dist > self.min_weight_dist and iter_num < self.max_iter:
            gr_w, gr_w0 = self._mserror_grad(X, Y)
            if iter_num == 0:
                # Чтобы eta адаптировалась к порядку градиента, делим на l2 норму градиента в нуле
                eta = self.eta / np.sqrt(np.linalg.norm(gr_w) ** 2 + gr_w0 ** 2)
            new_w = self.w - eta * gr_w
            new_w0 = self.w0 - eta * gr_w0
            weight_dist = np.sqrt(np.linalg.norm(new_w - self.w) ** 2 + (new_w0 - self.w0) ** 2)
            iter_num += 1
            w_ls.append(new_w)
            w0_ls.append(new_w0)
            self.w = new_w
            self.w0 = new_w0
        self.w_ls = w_ls
        self.w0_ls = w0_ls

    def fit(self, X, Y):
        """
        Обучает модель
        :param X: известные атрибуты (фичи) объектов, по которым мы собираемся получить предсказание
        :param Y: известный целевой атрибут
        :return: класс LinearRegression с подобранными весами
        """
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        self._optimize(X, Y)

    def predict(self, X):
        """
        Возвращает предсказанный атрибут на основе коэффициентов, определенных в в результате обучения модели.
        :param X: Вектор или матрица атрибутов
        :return: class MyLinearRegression
        """
        return (X.dot(self.w.T)+self.w0).flatten()

    def test(self, X, Y):
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        return self._mserror(X, Y)
