try:
    from my_linear_regression import *
    from my_trees import *
    from korenblum_trees import *
    from quality_metrics import *
    from lesson_forest import *
except ImportError:
    from .my_linear_regressions import *
    from .my_trees import *
    from .korenblum_trees import *
    from .quality_metrics import *
    from .lesson_forest import *
