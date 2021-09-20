# -*- coding: UTF-8 -*-
"""
@Project ：kalman-filter-in-single-object-tracking 
@File ：matcher.py
@Date ：9/15/21 10:19 AM 
"""
import networkx as nx
import numpy as np
import utils


class Matcher:
    def __init__(self):
        pass

    @classmethod
    def match(cls, state_list, measure_list):  # 两个list中单个元素数据结构都是[c_x, c_y, w, h].T
        """
        最大权值匹配
        :param state_list: 先验状态list
        :param measure_list: 量测list
        :return: dict 匹配结果， eg:{'state_1': 'mea_1', 'state_0': 'mea_0'}
        """
        graph = nx.Graph()
        for idx_sta, state in enumerate(state_list):
            state_node = 'state_%d' % idx_sta
            graph.add_node(state_node, bipartite=0)
            for idx_mea, measure in enumerate(measure_list):
                mea_node = 'mea_%d' % idx_mea
                graph.add_node(mea_node, bipartite=1)
                score = cls.cal_iou(state, measure)
                if score is not None:
                    graph.add_edge(state_node, mea_node, weight=score)
        match_set = nx.max_weight_matching(graph)
        res = dict()
        for (node_1, node_2) in match_set:
            if node_1.split('_')[0] == 'mea':
                node_1, node_2 = node_2, node_1
            res[node_1] = node_2
        return res

    @classmethod
    def cal_iou(cls, state, measure):
        """
        计算状态和观测之间的IOU
        :param state:ndarray [c_x, c_y, w, h].T
        :param measure:ndarray [c_x, c_y, w, h].T
        :return:
        """
        state = utils.mea2box(state)  # [lt_x, lt_y, rb_x, rb_y].T
        measure = utils.mea2box(measure)  # [lt_x, lt_y, rb_x, rb_y].T
        s_tl_x, s_tl_y, s_br_x, s_br_y = state[0], state[1], state[2], state[3]
        m_tl_x, m_tl_y, m_br_x, m_br_y = measure[0], measure[1], measure[2], measure[3]
        # 计算相交部分的坐标
        x_min = max(s_tl_x, m_tl_x)
        x_max = min(s_br_x, m_br_x)
        y_min = max(s_tl_y, m_tl_y)
        y_max = min(s_br_y, m_br_y)
        inter_h = max(y_max - y_min + 1, 0)
        inter_w = max(x_max - x_min + 1, 0)
        inter = inter_h * inter_w
        if inter == 0:
            return 0
        else:
            return inter / ((s_br_x - s_tl_x) * (s_br_y - s_tl_y) + (m_br_x - m_tl_x) * (m_br_y - m_tl_y) - inter)


if __name__ == '__main__':
    # state_list = [np.array([10, 10, 5, 5]).T, np.array([30, 30, 5, 5]).T]
    state_list = []
    measure_list = [np.array([12, 12, 5, 5]).T, np.array([28, 28, 5, 5]).T]
    match_dict = Matcher.match(state_list, measure_list)
    print(match_dict)
    for state, mea in match_dict.items():
        print(state, mea)

