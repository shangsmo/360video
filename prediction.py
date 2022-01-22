from scipy import stats
from statsmodels.tsa.ar_model import AutoReg
import numpy as np

from nntrain import *


def linear_model(x_parameter, y_parameter, predict_value):
    predictions = []
    for i in range(len(y_parameter)):
        slope, intercept, r, p, std_err = stats.linregress(x_parameter, y_parameter[i])
        predictions.append([slope * j + intercept for j in predict_value])
    predictions = standardize(predictions)
    return predictions


def avg_model(x_parameter, y_parameter, predict_value):
    predictions = []
    for i in range(len(y_parameter)):
        mean = np.mean(y_parameter[i])
        predictions.append([0 * j + mean for j in predict_value])
    predictions = standardize(predictions)
    return predictions


def ar_model(x_parameter, y_parameter, predict_value):
    predictions = []
    for i in range(len(y_parameter)):
        model_fit = AutoReg(y_parameter[i], 5).fit()
        predictions.append(model_fit.predict(len(y_parameter[i]), len(y_parameter[i]) + len(predict_value) - 1))
    predictions = standardize(predictions)
    return predictions


def nnar_model(x_parameter, tr, net, all_users_trajectory):
    predictions = [[], []]
    input = get_input(x_parameter[0], all_users_trajectory, tr)
    tensor = net(Variable(torch.unsqueeze(torch.FloatTensor([input]), dim=1)))
    list = tensor.detach().numpy().tolist()[0][0]
    for i in range(len(list)):
        predictions[i % 2].append(list[i])
    predictions = standardize(predictions)
    return predictions


def standardize(predictions):
    for i in range(len(predictions[0])):
        tmp = predictions[0][i]
        if math.sin(predictions[0][i]) >= 0:
            predictions[0][i] = math.acos(math.cos(predictions[0][i]))
        else:
            predictions[0][i] = -math.acos(math.cos(predictions[0][i]))
    for i in range(len(predictions[1])):
        predictions[1][i] = math.asin(math.sin(predictions[1][i]))
    return predictions
