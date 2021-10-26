try:
    from my_linear_regression import MyLinearRegression
    from my_linear_regressions import MyGD
    from my_linear_regressions import MySGD
    from my_linear_regressions import MyMiniBatchGD
    from my_trees import MyBaseTree
except ImportError:
    from .my_linear_regressions import MyLinearRegression
    from .my_linear_regressions import MyGD
    from .my_linear_regressions import MySGD
    from .my_linear_regressions import MyMiniBatchGD
    from .my_trees import MyBaseTree