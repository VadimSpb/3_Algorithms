import numpy as np
import pandas as pd


class MyQualityMetrics:
    def __init__(self, y_real, y_pred):
        self.y_real = y_real
        self.y_pred = y_pred
        self.accuracy = self._accurancy
        self.tp = self._tp()
        self.fp = self._fp()
        self.tn = self._tn()
        self.fn = self._fn()
        self.matrix = self._correlation_matrix()
        self.accuracy = self._accurancy()
        self.precission = self._precission()
        self.recall = self._recall()
        self.f1score = self._f1score()
        self.stats = self._stats()

    def _tp(self):
        tp: int = 0
        for i in range(len(self.y_real)):
            if (self.y_real[i] == 1) & (self.y_pred[i] == 1):
                tp += 1
        return tp

    def _fp(self):
        fp: int = 0
        for i in range(len(self.y_real)):
            if (self.y_real[i] == 0) & (self.y_pred[i] == 1):
                fp += 1
        return fp

    def _tn(self):
        tn: int = 0
        for i in range(len(self.y_real)):
            if (self.y_real[i] == 0) & (self.y_pred[i] == 0):
                tn += 1
        return tn

    def _fn(self):
        fn = 0
        for i in range(len(self.y_real)):
            if (self.y_real[i] == 1) & (self.y_pred[i] == 0):
                fn += 1
        return fn

    def _correlation_matrix(self):
        return pd.DataFrame.from_dict(
            {'y_pred': ['a(x) = +1', 'a(x) = -1'],
             'y = +1': [self.tp, self.fn],
             'y= - 1': [self.fp, self.tn]}).set_index('y_pred')

    def _accurancy(self):
        true_pred = 0
        for i in range(len(self.y_real)):
            if self.y_pred[i] == self.y_real[i]:
                true_pred += 1
        return true_pred / len(self.y_real)

    def _precission(self):
        return self.tp / (self.tp + self.fp)

    def _recall(self):
        return self.tp / (self.tp + self.fn)

    def _f1score(self):
        return (2 * self.precission * self.recall /
                (self.precission + self.recall))

    def _stats(self):
        return pd.DataFrame(
            np.array([self.tp, self.tn, self.fn, self.fp,
                     self.accuracy, self.precission, self.recall, self.f1score]),
            index=['tp', 'tn', 'fn', 'fp', 'accuracy', 'precission', 'recall', 'f1score']
        )
