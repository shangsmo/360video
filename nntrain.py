import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = F.relu(out)
        out = self.hidden2(out)
        out = torch.sigmoid(out)
        out = self.predict(out)
        return out


def train(experimentNo, videoNo, all_users_trajectory):
    inputs = []
    outputs = []
    # 构建输入、输出
    for n in range(len(all_users_trajectory)):
        tr = all_users_trajectory[n]
        for i in range(len(tr.yaws) - 8):
            input = get_input(i, all_users_trajectory, tr, n)
            output = [tr.yaws[i + 4], tr.pitchs[i + 4], tr.yaws[i + 5], tr.pitchs[i + 5], tr.yaws[i + 6],
                      tr.pitchs[i + 6], tr.yaws[i + 7], tr.pitchs[i + 7]]

            inputs.append(input)
            outputs.append(output)
    x, y = (Variable(torch.unsqueeze(torch.FloatTensor(inputs), dim=1)),
            Variable(torch.unsqueeze(torch.FloatTensor(outputs), dim=1)))
    net = Net(17, 20, 8)
    print(net)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    for t in range(50000):
        prediction = net(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 500 == 0:
            print(str(t) + '_Loss = %.4f' % loss.data)
    torch.save(net.state_dict(), './nnar_model/NNAR_' + str(experimentNo) + '_' + str(videoNo) + '_' + str(
        int(loss.data * 10000)) + '.pth')
    return net


def get_rou(a, b):
    a_yaws = []
    a_pitchs = []
    b_yaws = []
    b_pitchs = []
    for i in range(len(a)):
        if i % 2 == 0:
            a_yaws.append(a[i])
            b_yaws.append(b[i])
        else:
            a_pitchs.append(a[i])
            b_pitchs.append(b[i])
    tmp = 0
    for i in range(len(a_yaws)):
        tmp += get_cos(a_yaws[i], a_pitchs[i], b_yaws[i], b_pitchs[i])
    rou = tmp / len(a_yaws)
    return rou


def get_cos(a_yaw, a_pitch, b_yaw, b_pitch):
    res = math.cos(a_yaw - b_yaw) * math.cos(a_pitch) * math.cos(b_pitch) + math.sin(a_pitch) * math.sin(b_pitch)
    return res


def get_input(beg, all_users_trajectory, tr, idx=1000):
    best_tr_no = -1  # 相关的轨迹编号
    rou = -1  # 相关系数
    input = [tr.yaws[beg], tr.pitchs[beg], tr.yaws[beg + 1], tr.pitchs[beg + 1], tr.yaws[beg + 2], tr.pitchs[beg + 2],
             tr.yaws[beg + 3], tr.pitchs[beg + 3]]
    for m in range(len(all_users_trajectory)):
        if m != idx:
            current_tr = all_users_trajectory[m]
            compare_tr = [current_tr.yaws[beg], current_tr.pitchs[beg], current_tr.yaws[beg + 1],
                          current_tr.pitchs[beg + 1], current_tr.yaws[beg + 2], current_tr.pitchs[beg + 2],
                          current_tr.yaws[beg + 3], current_tr.pitchs[beg + 3]]
            current_rou = get_rou(input, compare_tr)
            if current_rou >= rou:
                best_tr_no = m
                rou = current_rou
    good_tr = all_users_trajectory[best_tr_no]
    input.extend([rou, good_tr.yaws[beg + 4], good_tr.pitchs[beg + 4], good_tr.yaws[beg + 5], good_tr.pitchs[beg + 5],
                  good_tr.yaws[beg + 6], good_tr.pitchs[beg + 6], good_tr.yaws[beg + 7], good_tr.pitchs[beg + 7]])
    return input
