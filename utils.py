import cv2
import numpy as np


def box2state(box):
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    w = box[2] - box[0]
    h = box[3] - box[1]
    return np.array([[center_x, center_y, w, h, 0, 0]]).T  # 定义为 [中心x,中心y,宽w,高h,dx,dy]


def state2box(state):
    center_x = state[0]
    center_y = state[1]
    w = state[2]
    h = state[3]
    return [int(i) for i in [center_x - w/2, center_y - h/2, center_x + w/2, center_y + h/2]]


def box2meas(box):
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    w = box[2] - box[0]
    h = box[3] - box[1]
    return np.array([[center_x, center_y, w, h]]).T  # 定义为 [中心x,中心y,宽w,高h]


def mea2box(mea):
    center_x = mea[0]
    center_y = mea[1]
    w = mea[2]
    h = mea[3]
    return [int(i) for i in [center_x - w/2, center_y - h/2, center_x + w/2, center_y + h/2]]


def mea2state(mea):
    return np.row_stack((mea, np.zeros((2, 1))))


def state2mea(state):
    return state.X_prior[0:4]


if __name__ == "__main__":
    # box1 = [100, 100, 200, 200]
    # box2 = [100, 100, 200, 300]
    # iou = cal_iou(box1, box2)
    # print(iou)
    # box1.pop(0)
    # box1.append(555)
    # print(box1)

    print(mea2state(np.array([[1,2,3,4]]).T))
