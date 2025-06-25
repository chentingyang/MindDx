import pickle

import matplotlib.pyplot as plt
from matplotlib import style

import numpy as np

import sbd
import util

# 联合诊断器
class diagnotor:
    def __init__(self, dataset, pattern_path, distribution_path):
        self.pl = pattern_lib(dataset, pattern_path)
        self.dl = distribution_lib(dataset, distribution_path)
        self.pl.load()
        self.dl.load()
        self.fault_num = len(dataset.anomaly_type)

    # 传入待诊断样本的窗口数据和z-score归一化后的窗口数据
    def confidence(self, ts, median, mad, dist):
        # ts = util.normalize([ts], 1)[0]
        pattern_conf = self.pl.confidence(ts, dist)
        # distribution_conf = self.dl.confidence(ts, median, mad)

        # pattern_rank = np.argsort(pattern_conf)5
        # distribution_rank = np.argsort(distribution_conf)
        # for i in range(len(pattern_rank)):
        #     pattern_conf[pattern_rank[i]] = i
        #     distribution_conf[distribution_rank[i]] = i

        # conf = np.array(pattern_conf) + np.array(distribution_conf)
        conf = np.array(pattern_conf)
        conf = conf.tolist()
        # print(pattern_conf)
        # print(distribution_conf)
        # print(conf)


        return conf


# 异常值分布库
class distribution_lib:
    def __init__(self, dataset, path):
        self.path = path
        self.fault_num = len(dataset.anomaly_type)
        # self.kpi_list = [1, 3, 4, 7, 8, 12, 13, 14, 15, 23, 30, 33, 69, 84, 105, 107, 109]
        self.kpi_list = [1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 29, 30, 32, 33, 69, 84, 103, 104,
                    105, 107, 109, 110]
        # self.kpi_list = [i for i in range(1, 85)] + [i for i in range(98, 111)]
        self.distribution_clusters = [[] for _ in range(self.fault_num)]
        self.dataset = dataset
        self.kpi_weight = []
        self.entropys = []
        self.ethreshold = 1.5
        self.zthreshold = 0.6
        self.zeros = []

    def load(self):
        for i in range(self.fault_num):
            for j in range(len(self.kpi_list)):
                clusters = pickle.load(open(self.path + r'\fault'+str(i+1)+r'\kpi_'+str(self.kpi_list[j]), 'rb'))
                self.distribution_clusters[i].append(clusters)
        for i in range(self.fault_num):
            zeros = []
            entropys = []
            for k in range(len(self.distribution_clusters[i])):
                clusters = self.distribution_clusters[i][k]['clusters']
                zero = self.distribution_clusters[i][k]['zero']
                cnum = []
                for c in clusters:
                    cnum.append(len(c))
                cnum.append(zero)
                entropys.append(util.calculate_entropy(cnum))
                zeros.append(zero / np.sum(cnum))
            self.zeros.append(zeros)
            self.entropys.append(entropys)
            # for j in range(len(entropys)):
            #     if entropys[i] > 0.5:
            #         entropys[i] = 1e10
            # self.kpi_weight.append(np.array(entropys)/np.sum(entropys))
            entropys = np.array(entropys)
            zeros = np.array(zeros)
            filter_entroys = entropys[entropys < self.ethreshold]
            filter_kpi_weight = util.back_softmax(filter_entroys)
            kpi_weight = [0] * len(self.kpi_list)
            j = 0
            for k in range(len(kpi_weight)):
                if entropys[k] < self.ethreshold:
                    kpi_weight[k] = filter_kpi_weight[j]
                    j += 1
            self.kpi_weight.append(np.array(kpi_weight))
            # self.kpi_weight.append(util.back_softmax(entropys))

        # for i in range(self.fault_num):
        #     print(self.kpi_weight[i].tolist())

    # 计算异常值分布置信度，返回一个向量存储每种异常类型的置信度
    def confidence(self, ts, median, mad):
        ret = []
        for i in range(self.fault_num):
            conf = []
            # entropys = []
            for k in range(len(self.distribution_clusters[i])):
                if self.kpi_weight[i][k] == 0:
                    conf.append(0)
                    continue
                clusters = self.distribution_clusters[i][k]['clusters']
                zero = self.distribution_clusters[i][k]['zero']
                distributions = self.distribution_clusters[i][k]['distributions']
                dist = []
                cnum = []
                x = ts[:, self.kpi_list[k]].flatten()
                x = (x - median[self.kpi_list[k]]) / mad[self.kpi_list[k]]
                cnt_up = 0
                cnt_low = 0
                for e in x:
                    # if self.kpi_list[k] == 84:
                    #     if e > 10:
                    #         cnt_up += 1
                    # else:
                    cdf = util.kpi_cdf3(e)
                    if cdf < 0.03:
                        cnt_low += 1
                    elif cdf > 0.97:
                        cnt_up += 1
                distribution = [cnt_low, len(x) - cnt_low - cnt_up, cnt_up]
                for c in clusters:
                    average_dist = 0
                    for t in c:
                        y = distributions[t]
                        average_dist += util.wasserstein_dist(distribution, y)
                    average_dist /= len(c)
                    dist.append(average_dist)
                    cnum.append(len(c))
                dist.append(util.wasserstein_dist(distribution, [0, 1, 0]))
                cnum.append(zero)
                weight = cnum / np.sum(cnum)
                # weight = util.sq_weight(cnum)

                # dist = np.square(dist)
                # dist = np.tanh((np.array(dist) - 0.3) * 10)

                conf.append(np.sum(dist * weight))
                # entropys.append(util.calculate_entropy(cnum))
                # print(dist)
            # print(conf)
            # kpi_weight = util.back_softmax(entropys)
            # print(entropys)
            # for l in range(len(conf)):
            #     if conf[l] < 0.3:
            #         conf[l] = 0
            #     else:
            #         conf[l] = 1
            # conf = np.tanh((np.array(conf)-0.3)*20)
            ret.append(np.sum(self.kpi_weight[i] * conf))
            # ret.append(np.sum(np.square(conf)))
            # ret.append(np.sum(conf) / self.fault_num)
        # print(ret)
        return ret


# 形状模式库
class pattern_lib:

    def __init__(self, dataset, path):
        self.path = path
        self.fault_num = len(dataset.anomaly_type)
        # self.kpi_list = [1, 3, 4, 7, 8, 12, 13, 14, 15, 23, 30, 33, 69, 84, 105, 107, 109]
        self.kpi_list = [1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 29, 30, 32, 33, 69, 84, 103, 104,
                    105, 107, 109, 110]
        # self.kpi_list = [i for i in range(1, 85)] + [i for i in range(98, 111)]
        self.pattern_clusters = [[] for _ in range(self.fault_num)]
        self.dataset = dataset
        self.penalty = 0.6
        self.kpi_weight = []
        self.entropys = []
        self.ethreshold = 1.5
        self.zthreshold = 0.6
        self.zeros = []

    def load(self):
        for i in range(self.fault_num):
            for j in range(len(self.kpi_list)):
                clusters = pickle.load(open(self.path + r'\fault'+str(i+1)+r'\kpi_'+str(self.kpi_list[j]), 'rb'))
                self.pattern_clusters[i].append(clusters)
        for i in range(self.fault_num):
            zeros = []
            entropys = []
            for k in range(len(self.pattern_clusters[i])):
                clusters = self.pattern_clusters[i][k]['clusters']
                zero = self.pattern_clusters[i][k]['zero']
                cnum = []
                for c in clusters:
                    cnum.append(len(c))
                cnum.append(zero)
                entropys.append(util.calculate_entropy(cnum))
                zeros.append(zero/np.sum(cnum))
            self.entropys.append(entropys)
            self.zeros.append(zeros)

            entropys = np.array(entropys)
            zeros = np.array(zeros)
            filter_entroys = entropys[entropys < self.ethreshold]
            filter_kpi_weight = util.back_softmax(filter_entroys)
            kpi_weight = [0] * len(self.kpi_list)
            j = 0
            for k in range(len(kpi_weight)):
                if entropys[k] < self.ethreshold:
                    kpi_weight[k] = filter_kpi_weight[j]
                    j += 1
            self.kpi_weight.append(np.array(kpi_weight))
            # self.kpi_weight.append(util.back_softmax(entropys))
        # for i in range(self.fault_num):
        #     print(self.kpi_weight[i].tolist())

    # 计算形状置信度，返回一个向量存储每种异常类型的置信度
    def confidence(self, ts, distribution):
        ret = []
        for i in range(self.fault_num):
            conf = []
            # entropys = []
            for k in range(len(self.pattern_clusters[i])):
                if self.kpi_weight[i][k] == 0:
                    conf.append(0)
                    continue
                clusters = self.pattern_clusters[i][k]['clusters']
                zero = self.pattern_clusters[i][k]['zero']
                dist = []
                cnum = []
                x = ts[:, self.kpi_list[k]].flatten()
                if not distribution[self.kpi_list[k]].cdf(np.var(x)) > 0.97:
                    for c in clusters:
                        average_dist = self.penalty
                        dist.append(average_dist)
                        cnum.append(len(c))
                    dist.append(0)
                    cnum.append(zero)
                else:
                    for c in clusters:
                        average_dist = 0
                        for t in c:
                            y = self.dataset.z_score_data[i][t][:, self.kpi_list[k]].flatten()
                            average_dist += sbd.sbd_distance(util.z_score_normalization(x), y)[0]
                        average_dist /= len(c)
                        dist.append(average_dist)
                        cnum.append(len(c))
                    dist.append(self.penalty)
                    cnum.append(zero)
                # weight = cnum / np.sum(cnum)
                weight = util.sq_weight(cnum)

                # dist = np.square(dist)
                # dist = np.tanh((np.array(dist) - 0.3) * 10)

                conf.append(np.sum(dist * weight))
                # entropys.append(util.calculate_entropy(cnum))
                # print(dist)
            # print(conf)
            # kpi_weight = util.back_softmax(entropys)
            # print(entropys)
            # for l in range(len(conf)):
            #     if conf[l] < 0.3:
            #         conf[l] = 0
            #     else:
            #         conf[l] = 1
            # conf = np.tanh((np.array(conf)-0.3)*20)
            ret.append(np.sum(self.kpi_weight[i] * conf))

            # ret.append(np.sum(np.square(conf)))
        # print(ret)
        return ret

