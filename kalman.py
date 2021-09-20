# -*- coding: UTF-8 -*-
"""
@Project ：kalman-filter-in-single-object-tracking 
@File ：kalman.py
@Date ：9/15/21 3:36 PM 
"""
import random
import numpy as np
import utils
from matcher import Matcher

GENERATE_SET = 1  # 设置航迹起始帧数
TERMINATE_SET = 7  # 设置航迹终止帧数


class Kalman:
    def __init__(self, A, B, H, Q, R, X, P):
        # 固定参数
        self.A = A  # 状态转移矩阵
        self.B = B  # 控制矩阵
        self.H = H  # 观测矩阵
        self.Q = Q  # 过程噪声
        self.R = R  # 量测噪声
        # 迭代参数
        self.X_posterior = X  # 后验状态, 定义为 [中心x,中心y,宽w,高h,dx,dy]
        self.P_posterior = P  # 后验误差矩阵
        self.X_prior = None  # 先验状态
        self.P_prior = None  # 先验误差矩阵
        self.K = None  # kalman gain
        self.Z = None  # 观测, 定义为 [中心x,中心y,宽w,高h]
        # 起始和终止策略
        self.terminate_count = TERMINATE_SET
        # 缓存航迹
        self.track = []  # 记录当前航迹[(p1_x,p1_y),()]
        self.track_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.__record_track()

    def predict(self):
        """
        预测外推
        :return:
        """
        self.X_prior = np.dot(self.A, self.X_posterior)
        self.P_prior = np.dot(np.dot(self.A, self.P_posterior), self.A.T) + self.Q
        return self.X_prior, self.P_prior

    @staticmethod
    def association(kalman_list, mea_list):
        """
        多目标关联，使用最大权重匹配
        :param kalman_list: 状态列表，存着每个kalman对象，已经完成预测外推
        :param mea_list: 量测列表，存着矩阵形式的目标量测 ndarray [c_x, c_y, w, h].T
        :return:
        """
        # 记录需要匹配的状态和量测
        state_rec = {i for i in range(len(kalman_list))}
        mea_rec = {i for i in range(len(mea_list))}

        # 将状态类进行转换便于统一匹配类型
        state_list = list()  # [c_x, c_y, w, h].T
        for kalman in kalman_list:
            state = kalman.X_prior
            state_list.append(state[0:4])

        # 进行匹配得到一个匹配字典
        match_dict = Matcher.match(state_list, mea_list)

        # 根据匹配字典，将匹配上的直接进行更新，没有匹配上的返回
        state_used = set()
        mea_used = set()
        match_list = list()
        for state, mea in match_dict.items():
            state_index = int(state.split('_')[1])
            mea_index = int(mea.split('_')[1])
            match_list.append([utils.state2box(state_list[state_index]), utils.mea2box(mea_list[mea_index])])
            kalman_list[state_index].update(mea_list[mea_index])
            state_used.add(state_index)
            mea_used.add(mea_index)

        # 求出未匹配状态和量测，返回
        return list(state_rec - state_used), list(mea_rec - mea_used), match_list

    def update(self, mea=None):
        """
        完成一次kalman滤波
        :param mea:
        :return:
        """
        status = True
        if mea is not None:  # 有关联量测匹配上
            self.Z = mea
            self.K = np.dot(np.dot(self.P_prior, self.H.T),
                            np.linalg.inv(np.dot(np.dot(self.H, self.P_prior), self.H.T) + self.R))  # 计算卡尔曼增益
            self.X_posterior = self.X_prior + np.dot(self.K, self.Z - np.dot(self.H, self.X_prior))  # 更新后验估计
            self.P_posterior = np.dot(np.eye(6) - np.dot(self.K, self.H), self.P_prior)  # 更新后验误差矩阵
            status = True
        else:  # 无关联量测匹配上
            if self.terminate_count == 1:
                status = False
            else:
                self.terminate_count -= 1
                self.X_posterior = self.X_prior
                self.P_posterior = self.P_prior
                status = True
        if status:
            self.__record_track()

        return status, self.X_posterior, self.P_posterior

    def __record_track(self):
        self.track.append([int(self.X_posterior[0]), int(self.X_posterior[1])])


if __name__ == '__main__':
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

    box = [729, 238, 764, 339]
    X = utils.box2state(box)

    k1 = Kalman(A, B, H, Q, R, X, P)
    print(k1.predict())

    mea = [730, 240, 766, 340]
    mea = utils.box2meas(mea)
    print(k1.update(mea))
