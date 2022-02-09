import matplotlib.pyplot as plt
import seaborn as sns

from process import *


def find_95th(errors):
    length = len(errors)
    key = int(length * 0.90)
    errors.sort()
    return errors[key]


def plot_error(yaw_error, pitch_error):
    time = []
    for i in range(len(yaw_error)):
        time.append(i * 0.5)
    fig, axes = plt.subplots(2,1, sharex=True)
    axes[0].plot(time, yaw_error, label=r'$\phi$')
    axes[0].set_ylabel(r'$\phi$ error(rad)')

    axes[1].plot(time, pitch_error, label=r'$\theta$')
    axes[1].set_ylabel(r'$\theta$ error(rad)')
    plt.xlabel('time(s)')
    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.savefig('./figure/error.png')
    plt.show()


def plot_pdf(error, lable=''):
    plt.subplot(221)
    sns.distplot(error[0], hist=False, kde=True, rug=True, label=lable)
    plt.legend()  # 显示图例
    plt.ylabel('')
    plt.title('0.5s')

    plt.subplot(222)
    sns.distplot(error[1], hist=False, kde=True, rug=True, label=lable)
    plt.legend()  # 显示图例
    plt.ylabel('')
    plt.title('1.0s')

    plt.subplot(223)
    sns.distplot(error[2], hist=False, kde=True, rug=True, label=lable)
    plt.legend()  # 显示图例
    plt.ylabel('')
    plt.title('1.5s')

    plt.subplot(224)
    sns.distplot(error[3], hist=False, kde=True, rug=True, label=lable)
    plt.legend()  # 显示图例
    plt.ylabel('')
    plt.title('2.0s')

    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 调整子图间距
    plt.savefig('./figure/pdf_pitch.png')
    plt.show()


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
    net.load_state_dict(torch.load('./nnar_model/NNAR_2_3_706.pth'))
    l = Trajectory(0.5, reader(2, 3, 46))
    all_users_trajectory = read_users(2, 3, users)
    error_05 = [[], [], [], [], []]
    error_10 = [[], [], [], [], []]
    error_15 = [[], [], [], [], []]
    error_20 = [[], [], [], [], []]

    yaw_error = [[], [], [], []]
    pitch_error = [[], [], [], []]

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

        yaw = prediction_nnar[0][step] - l.yaws[t + 4 + step]
        pitch = prediction_nnar[1][step] - l.pitchs[t + 4 + step]
        if yaw > math.pi:
            yaw = yaw - 2*math.pi
        elif yaw < -math.pi:
            yaw = 2*math.pi + yaw
        yaw_error[step].append(yaw)
        pitch_error[step].append(pitch)

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

        yaw = prediction_nnar[0][step] - l.yaws[t + 4 + step]
        pitch = prediction_nnar[1][step] - l.pitchs[t + 4 + step]
        if yaw > math.pi:
            yaw = yaw - 2*math.pi
        elif yaw < -math.pi:
            yaw = 2*math.pi + yaw
        yaw_error[step].append(yaw)
        pitch_error[step].append(pitch)

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

        yaw = prediction_nnar[0][step] - l.yaws[t + 4 + step]
        pitch = prediction_nnar[1][step] - l.pitchs[t + 4 + step]
        if yaw > math.pi:
            yaw = yaw - 2*math.pi
        elif yaw < -math.pi:
            yaw = 2*math.pi + yaw
        yaw_error[step].append(yaw)
        pitch_error[step].append(pitch)

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

        yaw = prediction_nnar[0][step] - l.yaws[t + 4 + step]
        pitch = prediction_nnar[1][step] - l.pitchs[t + 4 + step]
        if yaw > math.pi:
            yaw = yaw - 2*math.pi
        elif yaw < -math.pi:
            yaw = 2*math.pi + yaw
        yaw_error[step].append(yaw)
        pitch_error[step].append(pitch)

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
    print(error_95th[0][3])

    plot_error(yaw_error[3],pitch_error[3])
    # plot_bar(error_95th)
    # plot_pdf(pitch_error, lable=r'$\theta$')