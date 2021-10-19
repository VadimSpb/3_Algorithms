import numpy as np


class MyGD:
    def __init__(
            self,
            alpha: int = 1e-3,
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
        return (x - x.mean(0)) / x.std(0)

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
                w = w - self.alpha * ((1 / n * 2 * np.dot((y_pred - y), x))  + 2 * self.lambda_ * w)
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
        :return: mse
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
                    w = w - self.alpha * (1 / n * 2 * np.dot((batch_y_pred - batch_y), batch_x) + self.lambda_ * w / abs(w))
                elif penalty == 'l2':
                    w = w - self.alpha * (1 / n * 2 * np.dot((batch_y_pred - batch_y), batch_x) + 2 * self.lambda_ * w)
                else:
                    w = w - self.alpha * (1/n * 2 * np.dot((batch_y_pred - batch_y), batch_x))
                logs.append([i, w, mse_err])
        self.logs = logs
        if self.logs:
            self.err_logs = sum(self.logs, [])[2::3]
        self.w = w
