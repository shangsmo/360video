from element import *
from prediction import standardize


def kalman(yaw, pitch, pre_yaw, pre_pitch, pre_observe, encoder, timeId):
    tiles = Chunk(encoder, timeId)
    R_yaws = []
    R_pitchs = []
    R_yaws_pitchs = []
    for i in range(len(pre_observe.times)):
        R_yaws.append((pre_observe.yaws[i] - pre_yaw) * (pre_observe.yaws[i] - pre_yaw))
        R_pitchs.append((pre_observe.pitchs[i] - pre_pitch) * (pre_observe.pitchs[i] - pre_pitch))
        R_yaws_pitchs.append((pre_observe.pitchs[i] - pre_pitch) * (pre_observe.yaws[i] - pre_yaw))
    r_yaw = np.mean(R_yaws)
    r_pitch = np.mean(R_pitchs)
    r_yaw_pitch = np.mean(R_yaws_pitchs)
    cov = [[r_yaw, r_yaw_pitch],[r_yaw_pitch, r_pitch]]
    sample = 500
    data = np.random.multivariate_normal([yaw, pitch], cov, sample)
    data = standardize(data)
    for i in range(sample):
        row, col = angle_to_tile(data[i][0], data[i][1], encoder)
        tiles.probabilitys[row][col] += 1/sample
    return tiles


def multi_user(all_trajectory, encoder, timeId):
    tiles = Chunk(encoder, timeId)
    num = 0
    for usr in all_trajectory:
        for i in range(len(usr)):
            row, col = angle_to_tile(usr[i][0], usr[i][1], encoder)
            tiles.probabilitys[row][col] += 1
            num += 1
    tiles.probabilitys = (np.array(tiles.probabilitys)/num).tolist()
    return tiles


def MLE(all_trajectory, yaw, pitch, pre_yaw, pre_pitch, pre_observe, encoder, timeId, before_kalman, before_multi):
    tiles = Chunk(encoder, timeId)
    tiles_kal = kalman(yaw, pitch, pre_yaw, pre_pitch, pre_observe, encoder, timeId)
    tiles_multi = multi_user(all_trajectory, encoder, timeId)
    star_alpha = 0
    star_res = 0
    for alpha in np.arange(0,1,0.01):
        tmp = alpha * np.array(before_kalman.probabilitys) + (1 - alpha) * np.array(before_multi.probabilitys)
        res = 1
        for i in range(len(pre_observe.times)):
            row, col = angle_to_tile(pre_observe.yaws[i], pre_observe.pitchs[i], encoder)
            res = res * tmp[row][col]
        if star_res < res:
            star_res = res
            star_alpha = alpha
    tiles.probabilitys = (star_alpha * np.array(tiles_kal.probabilitys) + (1 - star_alpha) * np.array(tiles_multi.probabilitys)).tolist()
    return tiles


def probability_to_weight(encoder, timeId):
    tiles = Chunk(encoder, timeId)
    return tiles


def angle_to_tile(yaw, pitch, encoder):
    yaw = yaw + math.pi
    pitch = pitch + math.pi/2
    tile_row_degree = 2*math.pi/encoder.lengthNum
    tile_col_degree = math.pi/encoder.hightNum
    col = int(yaw/tile_row_degree)
    row = int(pitch/tile_col_degree)
    return row, col