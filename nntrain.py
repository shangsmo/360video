import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from process import *

ips = 0
ops = 0
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

def train(experimentNo, videoNo, users):
    inputs = []
    outputs = []
    all_users_trajectory = []
    # 初始化所有训练轨迹
    for user in users:
        org = reader(experimentNo, videoNo, user)
        trajectory = Trajectory(0.5, org)
        all_users_trajectory.append(trajectory)
    # 构建输入、输出
    for n in range(len(all_users_trajectory)):
        tr = all_users_trajectory[n]
        for i in range(len(tr.yaws) - 7):
            input = [tr.yaws[i], tr.pitchs[i], tr.yaws[i + 1], tr.pitchs[i + 1], tr.yaws[i + 2], tr.pitchs[i + 2],
                     tr.yaws[i + 3], tr.pitchs[i + 3]]
            output = [tr.yaws[i + 4], tr.pitchs[i + 4], tr.yaws[i + 5], tr.pitchs[i + 5], tr.yaws[i + 6],
                      tr.pitchs[i + 6], tr.yaws[i + 7], tr.pitchs[i + 7]]
            # 寻找最相关轨迹
            best_tr_no = -1  # 相关的轨迹编号
            rou = -1  # 相关系数
            for m in range(len(all_users_trajectory)):
                if m != n:
                    current_tr = all_users_trajectory[m]
                    compare_tr = [current_tr.yaws[i], current_tr.pitchs[i], current_tr.yaws[i + 1],
                                  current_tr.pitchs[i + 1], current_tr.yaws[i + 2], current_tr.pitchs[i + 2],
                                  current_tr.yaws[i + 3], current_tr.pitchs[i + 3]]
                    current_rou = getRou(input, compare_tr)
                    if current_rou >= rou:
                        best_tr_no = m
                        rou = current_rou
            print(best_tr_no)
            print(rou)
            good_tr = all_users_trajectory[best_tr_no]
            input.extend([rou, good_tr.yaws[i + 4], good_tr.pitchs[i + 4], good_tr.yaws[i + 5], good_tr.pitchs[i + 5],
                          good_tr.yaws[i + 6], good_tr.pitchs[i + 6], good_tr.yaws[i + 7], good_tr.pitchs[i + 7]])

            inputs.append(input)
            outputs.append(output)
    x, y = (Variable(torch.unsqueeze(torch.FloatTensor(inputs), dim=1)), Variable(torch.unsqueeze(torch.FloatTensor(outputs), dim=1)))
    net = Net(17, 20, 8)
    print(net)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    loss_func = torch.nn.MSELoss()

    for t in range(50000):
        prediction = net(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 5 == 0:
            print('Loss = %.4f' % loss.data)
    torch.save(net,'./nnar_model/NNAR_'+str(experimentNo)+'_'+str(videoNo)+'.pth')
    return net

def getRou(tr, compare):
    tr_yaws = []
    tr_pitchs = []
    compare_yaws = []
    compare_pitchs = []
    for i in range(len(tr)):
        if i%2 == 0:
            tr_yaws.append(tr[i])
            compare_yaws.append(compare[i])
        else:
            tr_pitchs.append(tr[i])
            compare_pitchs.append(compare[i])
    tr_yaws = np.array(tr_yaws)
    compare_yaws = np.array(compare_yaws)
    tr_pitchs = np.array(tr_pitchs)
    compare_pitchs = np.array(compare_pitchs)
    rou_yaw = math.pow(np.inner(tr_yaws,compare_yaws),2)/(np.inner(tr_yaws,tr_yaws) * np.inner(compare_yaws,compare_yaws))
    rou_pitch = math.pow(np.inner(tr_pitchs, compare_pitchs),2)/(np.inner(tr_pitchs, tr_pitchs) * np.inner(compare_pitchs, compare_pitchs))
    rou = math.sqrt((rou_yaw + rou_pitch)/2)
    return rou

if __name__ == '__main__':
    users = []
    for i in range(42):
        users.append(i + 1)
    for group in [1]:
        net = train(group,4,users)



