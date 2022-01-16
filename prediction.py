import numpy as np
from scipy import stats
from statsmodels.tsa.ar_model import AutoReg


def linear_model(x_parameter, y_parameter, predict_value):
    slope, intercept, r, p, std_err = stats.linregress(x_parameter, y_parameter)
    predictions = [slope * i + intercept for i in predict_value]
    print(predictions)
    return predictions


def avg_model(x_parameter, y_parameter, predict_value):
    mean = np.mean(y_parameter)
    predictions = [0 * i + mean for i in predict_value]
    print(predictions)
    return predictions


def ar_model(x_parameter, y_parameter, predict_value):
    train = y_parameter
    model_fit = AutoReg(train,20).fit()
    predictions = model_fit.predict(len(train),len(train)+len(predict_value)-1)
    print(predictions)
    return predictions

def nnar_model(x_parameter, y_parameter, net):
    y_parameter