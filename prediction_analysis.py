import matplotlib.pyplot as plt

from process import *


def find_95th(errors):
    length = len(errors)
    key = int(length * 0.90)
    errors.sort()
    return errors[key]


def plot_bar(data):
    times = ('0.5s', '1.0s', '1.5s', '2.0s')
    bar_width = 0.1  # 条形宽度
    index_nnar = np.arange(len(times))
    index_lr = index_nnar + bar_width
    index_avg = index_lr + bar_width
    index_naive = index_avg + bar_width
    index_ar = index_naive + bar_width
    plt.bar(index_nnar, height=data[0], width=bar_width, label='NNAR')
    plt.bar(index_lr, height=data[1], width=bar_width, label='LR')
    plt.bar(index_avg, height=data[2], width=bar_width, label='Avg')
    plt.bar(index_naive, height=data[3], width=bar_width, label='Naive')
    plt.bar(index_ar, height=data[4], width=bar_width, label='AR')

    plt.legend()  # 显示图例
    plt.xticks(index_lr + bar_width / 2, times)
    plt.ylabel('p95(rad)')
    plt.title('Hey VR Interview')
    plt.savefig('./figure/p95_Hey_VR_Interview.png')
    plt.show()


if __name__ == '__main__':
    init()
    users = []
    for i in range(40):
        users.append(i + 1)
    net = Net(17, 20, 8)
    net.load_state_dict(torch.load('./nnar_model/NNAR_2_8_518.pth'))
    l = Trajectory(0.5, reader(2, 8, 46))
    all_users_trajectory = read_users(2, 8, users)
    error_05 = [[], [], [], [], []]
    error_10 = [[], [], [], [], []]
    error_15 = [[], [], [], [], []]
    error_20 = [[], [], [], [], []]
    for t in range(16, len(l.yaws) - 8):
        prediction_nnar = nnar_model([t], l, net, all_users_trajectory)
        prediction_linear = linear_model([t, t + 1, t + 2, t + 3], [l.yaws[t:t + 4], l.pitchs[t:t + 4]],
                                         [t + 4, t + 5, t + 6, t + 7])
        prediction_avg = avg_model([t, t + 1, t + 2, t + 3], [l.yaws[t:t + 4], l.pitchs[t:t + 4]],
                                   [t + 4, t + 5, t + 6, t + 7])
        prediction_naive = avg_model([t + 2, t + 3], [l.yaws[t + 2:t + 4], l.pitchs[t + 2:t + 4]],
                                     [t + 4, t + 5, t + 6, t + 7])
        prediction_ar = ar_model([t], [l.yaws[t - 16:t + 4], l.pitchs[t - 16:t + 4]], [t + 4, t + 5, t + 6, t + 7])
        step = 0
        error_05[0].append(math.acos(
            get_cos(prediction_nnar[0][step], prediction_nnar[1][step], l.yaws[t + 4 + step], l.pitchs[t + 4 + step])))
        error_05[1].append(math.acos(
            get_cos(prediction_linear[0][step], prediction_linear[1][step], l.yaws[t + 4 + step],
                    l.pitchs[t + 4 + step])))
        error_05[2].append(math.acos(
            get_cos(prediction_avg[0][step], prediction_avg[1][step], l.yaws[t + 4 + step], l.pitchs[t + 4 + step])))
        error_05[3].append(math.acos(get_cos(prediction_naive[0][step], prediction_naive[1][step], l.yaws[t + 4 + step],
                                             l.pitchs[t + 4 + step])))
        error_05[4].append(math.acos(
            get_cos(prediction_ar[0][step], prediction_ar[1][step], l.yaws[t + 4 + step], l.pitchs[t + 4 + step])))
        step = 1
        error_10[0].append(math.acos(
            get_cos(prediction_nnar[0][step], prediction_nnar[1][step], l.yaws[t + 4 + step], l.pitchs[t + 4 + step])))
        error_10[1].append(math.acos(
            get_cos(prediction_linear[0][step], prediction_linear[1][step], l.yaws[t + 4 + step],
                    l.pitchs[t + 4 + step])))
        error_10[2].append(math.acos(
            get_cos(prediction_avg[0][step], prediction_avg[1][step], l.yaws[t + 4 + step], l.pitchs[t + 4 + step])))
        error_10[3].append(math.acos(get_cos(prediction_naive[0][step], prediction_naive[1][step], l.yaws[t + 4 + step],
                                             l.pitchs[t + 4 + step])))
        error_10[4].append(math.acos(
            get_cos(prediction_ar[0][step], prediction_ar[1][step], l.yaws[t + 4 + step], l.pitchs[t + 4 + step])))
        step = 2
        error_15[0].append(math.acos(
            get_cos(prediction_nnar[0][step], prediction_nnar[1][step], l.yaws[t + 4 + step], l.pitchs[t + 4 + step])))
        error_15[1].append(math.acos(
            get_cos(prediction_linear[0][step], prediction_linear[1][step], l.yaws[t + 4 + step],
                    l.pitchs[t + 4 + step])))
        error_15[2].append(math.acos(
            get_cos(prediction_avg[0][step], prediction_avg[1][step], l.yaws[t + 4 + step], l.pitchs[t + 4 + step])))
        error_15[3].append(math.acos(get_cos(prediction_naive[0][step], prediction_naive[1][step], l.yaws[t + 4 + step],
                                             l.pitchs[t + 4 + step])))
        error_15[4].append(math.acos(
            get_cos(prediction_ar[0][step], prediction_ar[1][step], l.yaws[t + 4 + step], l.pitchs[t + 4 + step])))
        step = 3
        error_20[0].append(math.acos(
            get_cos(prediction_nnar[0][step], prediction_nnar[1][step], l.yaws[t + 4 + step], l.pitchs[t + 4 + step])))
        error_20[1].append(math.acos(
            get_cos(prediction_linear[0][step], prediction_linear[1][step], l.yaws[t + 4 + step],
                    l.pitchs[t + 4 + step])))
        error_20[2].append(math.acos(
            get_cos(prediction_avg[0][step], prediction_avg[1][step], l.yaws[t + 4 + step], l.pitchs[t + 4 + step])))
        error_20[3].append(math.acos(get_cos(prediction_naive[0][step], prediction_naive[1][step], l.yaws[t + 4 + step],
                                             l.pitchs[t + 4 + step])))
        error_20[4].append(math.acos(
            get_cos(prediction_ar[0][step], prediction_ar[1][step], l.yaws[t + 4 + step], l.pitchs[t + 4 + step])))

    error_mean = [[], [], [], [], []]
    error_95th = [[], [], [], [], []]
    for i in range(5):
        error_mean[i].append(np.mean(error_05[i]))
        error_mean[i].append(np.mean(error_10[i]))
        error_mean[i].append(np.mean(error_15[i]))
        error_mean[i].append(np.mean(error_20[i]))
        error_95th[i].append(find_95th(error_05[i]))
        error_95th[i].append(find_95th(error_10[i]))
        error_95th[i].append(find_95th(error_15[i]))
        error_95th[i].append(find_95th(error_20[i]))

    plot_bar(error_95th)
