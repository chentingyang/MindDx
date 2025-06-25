import pickle
import random

import numpy as np
from scipy.stats import norm

import util


class dataset:
    def __init__(self, path):
        self.path = path
        # 存无异常的数据
        self.normal_data = []
        # 存异常发生时间段
        self.labels = []
        # 原始KPI数据
        self.raw_data = []
        # 划分窗口后的KPI数据
        self.aligned_data = []
        # 对窗口数据进行z-score归一化后的数据
        self.z_score_data = []
        # 对窗口数据进行min-max归一化后的数据
        self.min_max_data = []
        # 异常类型的标识
        self.anomaly_type = ['fault1', 'fault2', 'fault3', 'fault4', 'fault5', 'lockwait', 'multiindex', 'setknob', 'stress']
        self.kpi_list = []
        self.median = {}
        self.mad = {}
        self.dist = {}


    # 从文件中加载一种异常类型的数据
    def load_file(self, path):
        f = open(path, 'rb')
        data = pickle.load(f)
        ts = []
        label = []
        for i in range(0, len(data)):
            ts.append(np.array(data[i][0]))
            label.append(data[i][1])
        return ts, label

    # 将整个数据集从文件中加载进来
    def load_data(self):
        self.normal_data = self.load_file(self.path + r'\normal_data.pickle')[0]
        for t in self.anomaly_type:
            anomaly_data, label = self.load_file(self.path + '\\' + t + r'_data.pickle')
            self.raw_data.append(anomaly_data)
            self.labels.append(label)
            # self.z_score_data.append(util.normalize(anomaly_data, 1))
            # self.min_max_data.append(util.normalize(anomaly_data, 2))

    def preprocess(self, kpi_list):
        self.align_data()
        self.model_normal(kpi_list)

    def model_normal(self, kpi_list):
        self.kpi_list = kpi_list
        # 计算无异常情况下每个kpi的median和mad
        for k in kpi_list:
            ts = []
            for t in self.normal_data:
                # ts.append((t[:, kpi] - 1638.4)/1638.4)
                ts.append(t[:, k])
            self.median[k], self.mad[k] = util.calculate_median_mad(ts)
            if self.mad[k] == 0:
                self.mad[k] = 1

        for k in kpi_list:
            variance = []
            for t in self.normal_data:
                # ts.append((t[:, kpi] - 1638.4)/1638.4)
                variance.append(np.var(t[:, k]))
            # 拟合分布
            # 使用scipy.stats.norm.fit来计算最适合数据的正态分布参数(均值，标准差)
            params = norm.fit(variance)
            # 创建一个正态分布的对象
            self.dist[k] = norm(*params)

    # 将原始数据划分时间窗口对齐
    def align_data(self):
        for i in range(len(self.anomaly_type)):
            data = []
            if self.anomaly_type[i] == 'setknob':
                for ts in self.raw_data[i]:
                    s = 0
                    while s + 10 < len(ts):
                        data.append(ts[s + 1:s + 1 + 10])
                        s += 12
            elif self.anomaly_type[i] == 'multiindex':
                for ts in self.raw_data[i]:
                    data.append(ts[1:1 + 10])
            else:
                for j in range(len(self.raw_data[i])):
                    data.append(self.raw_data[i][j][self.labels[i][j][0] - 1:self.labels[i][j][0] - 1 + 10])
            self.aligned_data.append(data)
            self.z_score_data.append(util.normalize(data, 1))
            self.min_max_data.append(util.normalize(data, 2))

    # 划分训练和测试集，测试集比例为ratio
    def split(self, ratio):
        train = dataset(self.path)
        test = dataset(self.path)
        for i in range(len(self.anomaly_type)):
            perm = [r for r in range(len(self.raw_data[i]))]
            random.shuffle(perm)
            test_id = set(perm[:int(len(self.raw_data[i]) * ratio)])
            ts_train = []
            ts_test = []
            label_train = []
            label_test = []
            for j in range(len(self.raw_data[i])):
                if j in test_id:
                    ts_test.append(self.raw_data[i][j])
                    label_test.append(self.labels[i][j])
                else:
                    ts_train.append(self.raw_data[i][j])
                    label_train.append(self.labels[i][j])
            test.raw_data.append(ts_test)
            test.labels.append(label_test)
            train.raw_data.append(ts_train)
            train.labels.append(label_train)
        test.normal_data = self.normal_data
        train.normal_data = self.normal_data
        test.align_data()
        train.align_data()
        test.model_normal(self.kpi_list)
        train.model_normal(self.kpi_list)

        return train, test


class multidataset:
    def __init__(self, datasets):
        self.datasets = datasets
        #存储数据在dataset中的索引
        self.index = []
        # 存异常发生时间段
        self.labels = []
        # 原始KPI数据
        self.raw_data = []
        # 划分窗口后的KPI数据
        self.aligned_data = []
        # 对窗口数据进行z-score归一化后的数据
        self.z_score_data = []
        #所有异常类型全部数据
        self.full_z_score_data = []
        # 对窗口数据进行min-max归一化后的数据
        self.min_max_data = []
        # 样本的形状离散化特征
        self.pattern_feature = []
        # 样本的值异常特征
        self.value_feature = []
        # 异常类型的标识
        self.anomaly_type = ['fault1', 'fault2', 'fault3', 'fault4', 'fault5', 'lockwait', 'multiindex', 'setknob',
                             'stress']

    def load(self):
        for i in range(len(self.anomaly_type)):
            index = []
            labels = []
            raw_data = []
            aligned_data = []
            z_score_data = []
            min_max_data = []
            for j in range(len(self.datasets)):
                index.extend([j] * len(self.datasets[j].aligned_data[i]))
                labels.extend(self.datasets[j].labels[i])
                raw_data.extend(self.datasets[j].raw_data[i])
                aligned_data.extend(self.datasets[j].aligned_data[i])
                z_score_data.extend(self.datasets[j].z_score_data[i])
                min_max_data.extend(self.datasets[j].min_max_data[i])
                self.full_z_score_data.extend(self.datasets[j].z_score_data[i])
            self.index.append(index)
            self.labels.append(labels)
            self.raw_data.append(raw_data)
            self.aligned_data.append(aligned_data)
            self.z_score_data.append(z_score_data)
            self.min_max_data.append(min_max_data)

