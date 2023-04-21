
import numpy as np

# target_data = [[位置（ｘ）,位置（ｙ）, 位置（ｚ）,
#                 速度（ｘ）,速度（ｙ）, 速度（ｚ）,
#                 旋回角度, パルス放射有無（0or1）,
#                 パルス放射方向（飛行方向との差、現状は全て0）,
#                 state(0), ..., state(199) ]]


def make_fix_data(target_data, dim):
    fix_data = np.zeros(
        target_data.shape[0], target_data.shape[1], target_data.shape[2] - 1, target_data.shape[3])

    if dim == 2:
        ans_data = np.zeros(
            target_data.shape[0], target_data.shape[1], target_data.shape[2] - 1, 2)

        for one_bat_data in target_data:
            for idx, one_episode_data in enumerate(one_bat_data):
                fix_data[0][idx][:][:] = one_episode_data[:-1][:]
                ans_data[0][idx][:][:] = np.array(
                    one_episode_data)[:-1, [3, 4, 7]]
    if dim == 3:
        ans_data = np.zeros(
            target_data.shape[0], target_data.shape[1], target_data.shape[2] - 1, 3)

        for i, one_bat_data in enumerate(target_data):
            for idx, one_episode_data in enumerate(one_bat_data):
                fix_data[i][idx][:][:] = one_episode_data[:-1][:]
                ans_data[i][idx][:][:] = np.array(one_episode_data)[
                    :-1, [3, 4, 5, 7]]

    return fix_data, ans_data
