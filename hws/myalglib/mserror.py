def mserror(x, w, y_pred):
    y = x.dot(w)
    return (sum((y - y_pred) ** 2)) / len(y)
