import numpy as np
import pandas as pd


# noinspection PyUnresolvedReferences
class MyGD:
    def __init__(
            self,
            alpha: float = 1e-3,
            random_seed: int = None,
            iter_num: int = 1e+4,
            scaler: str = None,
            lambda_: float = 1e-8
    ):

        self.alpha = alpha
        self.random_seed = random_seed
        self.iter_num = int(iter_num)
        self.logs = None
        self.err_logs = None
        self.w = None
        self.scaler = scaler
        self.lambda_ = lambda_
        self.x = None
        self.y = None
        self.y_pred = None

    @staticmethod
    def _mserror(y, y_pred):
        """
        Returns mean square error
        :param y: real target
        :param y_pred: predicted target
        :return: mse
        """
        return np.mean((y - y_pred) ** 2)

    @staticmethod
    def _normaliser(x):
        return (x - x.min(0)) / (x.max(0) - x.min(0))

    @staticmethod
    def _standardizer(x):
        return (x - x.mean()) / x.std()

    def fit(self, x, y, penalty=None):
        """
        Returns mean square error
        :param x: features
        :param y: real target
        :return: mse
        """
        if self.random_seed:
            np.random.seed(self.random_seed)
        logs = []
        w = np.random.randn(x.shape[1])
        n = x.shape[0]
        if self.scaler == 'normal':
            x = self._normaliser(x)
        if self.scaler == 'standart':
            x = self._standardizer(x)
        for i in range(self.iter_num):
            y_pred = np.dot(x, w)
            mse_err = self._mserror(y, y_pred)
            if penalty == 'l1':
                w = w - self.alpha * ((1 / n * 2 * np.dot((y_pred - y), x)) + self.lambda_ * w / abs(w))
            elif penalty == 'l2':
                w = w - self.alpha * ((1 / n * 2 * np.dot((y_pred - y), x)) + 2 * self.lambda_ * w)
            else:
                w = w - self.alpha * (1 / n * 2 * np.dot((y_pred - y), x))
            logs.append([i, w, mse_err])
        self.logs = logs
        if self.logs:
            self.err_logs = sum(self.logs, [])[2::3]
        self.w = w

    def predict(self, x):
        """
        predict by fitted model
        """
        return x.dot(self.w.T).flatten()


class MySGD(MyGD):
    def fit(self, x, y, penalty=None):
        """
        Returns mean square error
        :param x: features
        :param y: real target
        :return: mse
        """
        if self.random_seed:
            np.random.seed(self.random_seed)
        logs = []
        w = np.random.randn(x.shape[1])
        n = x.shape[0]
        if self.scaler == 'normal':
            x = self._normaliser(x)
        if self.scaler == 'standart':
            x = self._standardizer(x)
        for i in range(self.iter_num):
            ind = np.random.randint(n)
            y_pred = np.dot(x[ind], w)
            mse_err = self._mserror(y[ind], y_pred)
            if penalty == 'l1':
                w = w - self.alpha * (1 / n * 2 * np.dot((y_pred - y[ind]), x[ind]) + self.lambda_ * w / abs(w))
            elif penalty == 'l2':
                w = w - self.alpha * (1 / n * 2 * np.dot((y_pred - y[ind]), x[ind]) + 2 * self.lambda_ * w)
            else:
                w = w - self.alpha * (1 / n * 2 * np.dot((y_pred - y[ind]), x[ind]))
            logs.append([i, w, mse_err])
        self.logs = logs
        if self.logs:
            self.err_logs = sum(self.logs, [])[2::3]
        self.w = w


class MyMiniBatchGD(MyGD):

    def __init__(self, qty_in_batch=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qty_in_batch = qty_in_batch

    def fit(self, x, y, penalty=None):
        """
        Returns mean square error
        :param x: features
        :param y: real target
        :return: mini-batch model
        """
        if self.random_seed:
            np.random.seed(self.random_seed)
        logs = []
        w = np.random.randn(x.shape[1])
        n = x.shape[0]
        n_batch = n // self.qty_in_batch
        if n % self.qty_in_batch:
            n_batch += 1
        if self.scaler == 'normal':
            x = self._normaliser(x)
        if self.scaler == 'standart':
            x = self._standardizer(x)
        for i in range(self.iter_num):
            for j in range(n_batch):
                start_ = self.qty_in_batch * j
                end_ = self.qty_in_batch * (j + 1)
                batch_x = x[start_: end_, :]
                batch_y = y[start_: end_]
                batch_y_pred = np.dot(batch_x, w)
                mse_err = self._mserror(batch_y, batch_y_pred)
                if penalty == 'l1':
                    w = w - self.alpha * (
                                1 / n * 2 * np.dot((batch_y_pred - batch_y), batch_x) + self.lambda_ * w / abs(w))
                elif penalty == 'l2':
                    w = w - self.alpha * (1 / n * 2 * np.dot((batch_y_pred - batch_y), batch_x) + 2 * self.lambda_ * w)
                else:
                    w = w - self.alpha * (1 / n * 2 * np.dot((batch_y_pred - batch_y), batch_x))
                logs.append([i, w, mse_err])
        self.logs = logs
        if self.logs:
            self.err_logs = sum(self.logs, [])[2::3]
        self.w = w


class MyLogReg(MyGD):
    """
    Hand-made logistic Regression classifier.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accuracy = None
        self.err_matrix = None
        self.precision = None
        self.recall = None
        self.f1_score = None

    def _penalty_value(self, penalty, w):
        """
        @return: L1, L2 penalty value or None
        """
        penalty_value = 0
        if penalty == 'l1':
            penalty_value = self.lambda_ * w / abs(w)
        if penalty == 'l2':
            penalty_value = 2 * self.lambda_ * w
        return penalty_value

    @staticmethod
    def _sigmoid(z):
        """
        A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _calc_logloss(y, y_pred):
        max_ = 10e+10 / (10e+10 + 1)  # 0.9999(9)
        min_ = 1 - max_  # 1.000000082740371e-11
        y_pred = np.clip(y_pred, min_, max_)
        err = - np.mean(y * np.log(y_pred) + (1.0 - y) * np.log(1.0 - y_pred))
        err = np.sum(err)
        return err

    @staticmethod
    def correlation_matrix(y_pred, y):
        TP = len(y_pred[(y == y_pred) & (y_pred == 1)])
        FN = len(y_pred[(y != y_pred) & (y_pred == 0)])
        FP = len(y_pred[(y != y_pred) & (y_pred == 1)])
        TN = len(y_pred[(y == y_pred) & (y_pred == 0)])
        corr_matrix = pd.DataFrame.from_dict(
            {'y_pred': ['a(x) = +1', 'a(x) = -1'],
             'y = +1': [TP, FN],
             'y = -1': [FP, TN]}).set_index('y_pred')
        return corr_matrix

    def fit(self, x, y, penalty=None):
        """
        :param x: features
        :param y: real target
        :return: LogRegModel
        """
        logs = []
        w = np.random.randn(x.shape[1])
        n = x.shape[0]
        self.x = x
        self.y = y
        if self.scaler == 'normal':
            x = self._normaliser(x)
        if self.scaler == 'standart':
            x = self._standardizer(x)
        for i in range(self.iter_num):
            z = np.dot(x, w)
            y_pred = self._sigmoid(z)
            log_loss_err = self._calc_logloss(y, y_pred)
            penalty_value = self._penalty_value(penalty, w)
            w = w - self.alpha * (1 / n * np.dot(x.T, (y_pred - y)) + penalty_value)
            logs.append([i, w, log_loss_err])
        self.logs = logs
        if self.logs:
            self.err_logs = sum(self.logs, [])[2::3]
        self.w = w

    def predict_proba(self, x=None):
        """
        Расчёт вероятности
        """
        if x is None:
            x = self.x
        score = x.dot(self.w.T).flatten()
        return 1 / (1 + np.exp(-score))

    def predict(self, x, thr=0.5):
        proba = self.predict_proba(x)
        y_pred = np.zeros(proba.shape, dtype=bool)
        y_pred[proba > thr] = 1
        y_pred[proba <= thr] = 0
        self.y_pred = y_pred
        self.accuracy = len(y_pred[self.y == y_pred]) / len(y_pred)
        self.err_matrix = self.correlation_matrix(y_pred, self.y)
        self.precision = (len(y_pred[(self.y == y_pred) & (y_pred == 1)]) /
                          (len(y_pred[(self.y == y_pred) & (y_pred == 1)]) +
                          len(y_pred[(self.y != y_pred) & (y_pred == 1)])))
        self.recall = (len(y_pred[(self.y == y_pred) & (y_pred == 1)]) /
                      (len(y_pred[(self.y == y_pred) & (y_pred == 1)]) +
                       len(y_pred[(self.y != y_pred) & (y_pred == 0)])))
        self.f1_score = (2 * self.precision * self.recall
                         / (self.precision + self.recall))
        return y_pred
