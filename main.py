# -*- coding: utf-8 -*-
"""
@Project: kalman-filter-in-single-object-tracking
@File   : main.py
@Author : Zhang P.H
@Date   : 2021/9/20
@Desc   :
"""
import cv2
import numpy as np
import const
import utils
import measure
from kalman import Kalman

# --------------------------------Kalman参数---------------------------------------
# 状态转移矩阵，上一时刻的状态转移到当前时刻
A = np.array([[1, 0, 0, 0, 1, 0],
              [0, 1, 0, 0, 0, 1],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])
# 控制输入矩阵B
B = None
# 过程噪声协方差矩阵Q，p(w)~N(0,Q)，噪声来自真实世界中的不确定性,
# 在跟踪任务当中，过程噪声来自于目标移动的不确定性（突然加速、减速、转弯等）
Q = np.eye(A.shape[0]) * 0.1
# 状态观测矩阵
H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0]])
# 观测噪声协方差矩阵R，p(v)~N(0,R)
# 观测噪声来自于检测框丢失、重叠等
R = np.eye(H.shape[0]) * 1
# 状态估计协方差矩阵P初始化
P = np.eye(A.shape[0])
# -------------------------------------------------------------------------------


def main():
    # 1.载入视频和目标位置信息
    cap = cv2.VideoCapture(const.VIDEO_PATH)  # 穿插着视频是为了方便展示
    meas_list_all = measure.load_measurement(const.FILE_DIR)
    sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # opencv3.0
    video_writer = cv2.VideoWriter(const.VIDEO_OUTPUT_PATH, fourcc, const.FPS, sz, True)
    # 2. 逐帧滤波
    state_list = []  # 单帧目标状态信息，存kalman对象
    frame_cnt = 1
    for meas_list_frame in meas_list_all:
        # --------------------------------------------加载当帧图像------------------------------------
        ret, frame = cap.read()
        if not ret:
            break

        # ---------------------------------------Kalman Filter for multi-objects-------------------
        # 预测
        for target in state_list:
            target.predict()
        # 关联
        mea_list = [utils.box2meas(mea) for mea in meas_list_frame]
        state_rem_list, mea_rem_list, match_list = Kalman.association(state_list, mea_list)
        # 状态没匹配上的，更新一下，如果触发终止就删除
        state_del = list()
        for idx in state_rem_list:
            status, _, _ = state_list[idx].update()
            if not status:
                state_del.append(idx)
        state_list = [state_list[i] for i in range(len(state_list)) if i not in state_del]
        # 量测没匹配上的，作为新生目标进行航迹起始
        for idx in mea_rem_list:
            state_list.append(Kalman(A, B, H, Q, R, utils.mea2state(mea_list[idx]), P))

        # -----------------------------------------------可视化-----------------------------------
        # 显示所有mea到图像上
        for mea in meas_list_frame:
            cv2.rectangle(frame, tuple(mea[:2]), tuple(mea[2:]), const.COLOR_MEA, thickness=1)
        # 显示所有的state到图像上
        for kalman in state_list:
            pos = utils.state2box(kalman.X_posterior)
            cv2.rectangle(frame, tuple(pos[:2]), tuple(pos[2:]), const.COLOR_STA, thickness=2)
        # 将匹配关系画出来
        for item in match_list:
            cv2.line(frame, tuple(item[0][:2]), tuple(item[1][:2]), const.COLOR_MATCH, 3)
        # 绘制轨迹
        for kalman in state_list:
            tracks_list = kalman.track
            for idx in range(len(tracks_list) - 1):
                last_frame = tracks_list[idx]
                cur_frame = tracks_list[idx + 1]
                # print(last_frame, cur_frame)
                cv2.line(frame, last_frame, cur_frame, kalman.track_color, 2)

        cv2.putText(frame, str(frame_cnt), (0, 50), color=const.RED, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5)
        cv2.imshow('Demo', frame)
        cv2.imwrite("./image/{}.jpg".format(frame_cnt), frame)
        video_writer.write(frame)
        cv2.waitKey(100)  # 显示 1000 ms 即 1s 后消失
        frame_cnt += 1

    cap.release()
    cv2.destroyAllWindows()
    video_writer.release()


if __name__ == '__main__':
    main()
