
import math
import numpy as np
from sklearn.metrics import mean_squared_error

def get_rmse(y_true, y_pred):
    '''
    Compute
    '''
    return math.sqrt(mean_squared_error(y_true, y_pred))

def get_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100