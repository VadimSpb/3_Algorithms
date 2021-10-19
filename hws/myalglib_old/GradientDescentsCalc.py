import numpy as np
#from mserror import mserror

np.random.seed(1234)


class GradientDescentsCalc:
    """
    eta -  шаг градиентного спуска
    max_iter - максимальное число итераций
    min_weight_dist - критерий сходимости (разница весов, при которой алгоритм останавливается)
    weight_dist - начальная разница весов
    """

    eta = 0.01
    max_iter = 1e6
    min_weight_dist = 1e-8
    weight_dist = np.inf

    def __init__(self, data, target):
        """
        :param data: n-мерный массив наблюдений (матрица)
        :param target: одномерный массив результата (вектор)

        n_features - количество атрибутов наблюдения
        w -  нулевые начальные веса
        w_true -  вектор истинных весов
        normal_equation - вектор весов по нормальному уравнению линейной регрессии
        """
        self.data = data
        self.target = target

        self.n_features = data.shape[1]
        self.w = np.zeros(self.n_features)
        self.w_true = np.random.normal(size=(self.n_features,))

        self.normal_equation = np.linalg.solve((data.T).dot(data),
                                               (data.T).dot(target))

    def mserror(self, x, w, y_pred):
        # расчёт среднеквадратичной ошибки
        y = x.dot(w)
        return (sum((y - y_pred) ** 2)) / len(y)

    def gd(self):
        """
        errors  - список значений ошибок после каждой итерации
        iter_num - счетчик итераций
        :return: Gradient descent,  MSE
        """

        errors = []
        iter_num = 0
        w_list = [self.w.copy()]
        weight_dist = np.inf
        w = self.w
        X = self.data
        Y = self.target

        while weight_dist > self.min_weight_dist \
                and iter_num < self.max_iter:
            new_w = w - 2 * self.eta * np.dot(X.T, (np.dot(X, w) - Y)) / Y.shape[0]
            weight_dist = np.linalg.norm(new_w - w, ord=2)

            w_list.append(new_w.copy())
            errors.append(self.mserror(X, new_w, Y))

            iter_num += 1
            w = new_w

        w_list = np.array(w_list)
        return w_list, errors

    def sgd(self):
        errors = []
        iter_num = 0
        w_list = [self.w.copy()]
        weight_dist = np.inf
        w = self.w
        data = self.data
        target = self.target
        means = np.mean(self.data, axis=0)
        stds = np.std(self.data, axis=0)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i][j] = (data[i][j] - means[j]) / stds[j]


        # ход градиентного спуска
        while weight_dist > self.min_weight_dist \
                and iter_num < self.max_iter:
            # генерируем случайный индекс объекта выборки
            train_ind = np.random.randint(data.shape[0])

            new_w = w - 2 * self.eta * np.dot(data[train_ind].T, (np.dot(data[train_ind], w) - target[train_ind])) / \
                    target.shape[0]

            weight_dist = np.linalg.norm(new_w - w, ord=2)

            w_list.append(new_w.copy())
            errors.append(self.mserror(data, new_w, target))

            iter_num += 1
            w = new_w

        w_list = np.array(w_list)
        return w_list, errors


