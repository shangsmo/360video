
from scipy import stats
from statsmodels.tsa.ar_model import AutoReg

from nntrain import *


def linear_model(x_parameter, y_parameter, predict_value):
    predictions = []
    for i in range(len(y_parameter)):
        slope, intercept, r, p, std_err = stats.linregress(x_parameter, y_parameter[i])
        predictions.append([slope * j + intercept for j in predict_value])
    return predictions


def avg_model(x_parameter, y_parameter, predict_value):
    predictions = []
    for i in range(len(y_parameter)):
        mean = np.mean(y_parameter[i])
        predictions.append([0 * j + mean for j in predict_value])
    return predictions


def ar_model(x_parameter, y_parameter, predict_value):
    predictions = []
    for i in range(len(y_parameter)):
        train = y_parameter[0]
        model_fit = AutoReg(train,20).fit()
        predictions.append(model_fit.predict(len(train),len(train)+len(predict_value)-1))
    return predictions


def nnar_model(x_parameter, tr, net, all_users_trajectory):
    predictions = [[],[]]
    input = get_input(x_parameter[0], all_users_trajectory, tr)
    tensor = net(Variable(torch.unsqueeze(torch.FloatTensor([input]), dim=1)))
    list = tensor.detach().numpy().tolist()[0][0]
    for i in range(len(list)):
           predictions[i%2].append(list[i])
    print(predictions)