import numpy as np
from myalglib import MyLinearRegression
#%%

# Возьмем 2 признака и 1000 объектов
n_features = 2
n_objects = 1000

#%%

# сгенерируем вектор истинных весов
w_true = np.random.normal(size=(1, n_features ))

#%%

# сгенерируем матрицу X, вычислим Y с добавлением случайного шума
X = np.random.uniform(-7, 7, (n_objects, n_features))
Y = X.dot(w_true.T) + np.random.normal(0, 0.5, size=(n_objects, 1))

#%% md

## Обучаем линейную регрессию

#%%

shift = np.random.uniform(0, 100)
shift

#%%

Y_shift = Y+shift
Y_shift[:10]

#%%

lr= MyLinearRegression(eta=0.01, max_iter=1e4, min_weight_dist=1e-8)
lr.fit(X, Y_shift)
lr.test(X, Y_shift)

#%%

# Сравним истинные веса с расчётными:
lr.w, w_true