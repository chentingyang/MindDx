
import pickle

from sklearn.feature_selection import mutual_info_classif

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt

from matplotlib import style

import numpy as np

from scipy.stats import norm



import hierarchicalcluster
import sbd
import util

from dataset import dataset, multidataset



def draw(ts, label, metric, color):
    # 将'date'列设置为索引
    # ts.set_index(0, inplace=True)
    # 假设你想绘制第二列的数据，列的索引是1
    column_data0 = ts[:label[0] + 1, metric].flatten()
    column_data1 = ts[label[0]:label[1] + 1, metric].flatten()
    column_data2 = ts[label[1]:, metric].flatten()
    # 绘制折线图
    # 绘制时间序列图
    plt.plot([i for i in range(0, label[0] + 1)], column_data0, linestyle='--', color=color)
    plt.plot([i for i in range(label[0], label[1] + 1)], column_data1, color=color)
    plt.plot([i for i in range(label[1], len(ts))], column_data2, linestyle='--', color=color)
    # column_data0.plot(linestyle='--', color=color)
    # column_data1.plot(color=color)
    # column_data2.plot(linestyle='--', color=color)


def triple_draw(data, label, metric, index):
    draw(data[index * 3], label[index * 3], metric, 'b')
    draw(data[index * 3 + 1], label[index * 3 + 1], metric, 'g')
    draw(data[index * 3 + 2], label[index * 3 + 2], metric, 'r')
    # print(sbd.sbd_distance(data[index * 3][:, metric].flatten()[:25], data[index * 3 + 1][:, metric].flatten()[:25]))
    # print(sbd.sbd_distance(data[index * 3][:, metric].flatten()[:25], data[index * 3 + 2][:, metric].flatten()[:25]))
    # print(
    #     sbd.sbd_distance(data[index * 3 + 1][:, metric].flatten()[:25], data[index * 3 + 2][:, metric].flatten()[:25]))


def draw_normal(ts, metric, color):
    # 将'date'列设置为索引
    # ts.set_index(0, inplace=True)
    # 假设你想绘制第二列的数据，列的索引是1
    column_data = ts[:, metric].flatten()
    # column_data.index = [(i + 6) for i in range(0, len(column_data))]
    style.use('ggplot')
    # 绘制折线图
    # 绘制时间序列图
    plt.plot(column_data)


# 绘制kpi曲线
def triple_draw_normal(data, metric, index):
    draw_normal(data[index * 3], metric, 'b')
    draw_normal(data[index * 3 + 1], metric, 'g')
    draw_normal(data[index * 3 + 2], metric, 'r')



# KPI的所有异常类型样本统一聚类
def cluster_per_kpi(data, metric, path):
    ts = []
    index = []
    zero_cluster = 0
    for fault in range(len(data.anomaly_type)):
        for i in range(len(data.z_score_data[fault])):
            if data.datasets[data.index[fault][i]].dist[metric].cdf(np.var(data.aligned_data[fault][i][:, metric])) > 0.97:
                index.append(len(ts) + zero_cluster)
                ts.append(data.z_score_data[fault][i][:, metric].flatten())
            else:
                zero_cluster += 1
    ret = hierarchicalcluster.hierarchical_clustering(ts, 1, 0.4)
    print("Total:" + str(len(ts) + zero_cluster))
    # ts = np.array(ts).transpose()
    # if len(ts.shape) > 1:
    #     for i in range(ts.shape[1]):
    #         plt.plot(ts[:, i])
    #     plt.show()
    cluster = -1
    for i in ret:
        if i is not None and i > cluster:
            cluster = i
    print('Total cluster: ' + str(cluster + 1))
    clusters = [[] for _ in range(cluster + 1)]
    for i in range(len(ret)):
        if ret[i] is not None:
            clusters[ret[i]].append(index[i])
    # for i in range(len(clusters)):
    #     if len(clusters[i]) > 5:
    #         print('cluster ' + str(i) + ': ' + str(len(clusters[i])))
    #         for j in clusters[i]:
    #             plt.plot(ts[:, j])
    #         plt.show()
    # print('cluster  zero: ' + str(zero_cluster))
    pickle.dump({'clusters': clusters, 'zero': zero_cluster},
                open(path + r'\kpi_' + str(metric), 'wb'))


#建模聚类分布
def analyze_clusters(data, kpi_list, path):
    pattern_lib = {}
    for k in kpi_list:
        pattern_lib[k] = pickle.load(open(path + r'\kpi_' + str(k), 'rb'))
    z_score_data = data.full_z_score_data
    for i in range(len(kpi_list)):
        k = kpi_list[i]
        clusters = pattern_lib[k]['clusters']
        cluster_distribution = []
        for j in range(len(clusters)):
            norm_dist = []
            dist_table = np.zeros((len(clusters[j]), len(clusters[j])))
            dist_table.fill(np.inf)
            np.fill_diagonal(dist_table, 0)
            for ci in range(len(clusters[j])):
                for cj in range(ci + 1, len(clusters[j])):
                    dist_table[ci,cj] = sbd.sbd_distance(z_score_data[clusters[j][ci]][:,k].flatten(), z_score_data[clusters[j][cj]][:,k].flatten())[0]
                    dist_table[cj,ci] = dist_table[ci,cj]
            for ci in range(len(clusters[j])):
                dist = 0
                for cj in range(len(clusters[j])):
                    dist += dist_table[ci,cj]
                norm_dist.append(util.sbd_to_norm(dist / len(clusters[j])))
            # for e in clusters[j]:
            #     dist = 0
            #     for ee in clusters[j]:
            #         dist += sbd.sbd_distance(z_score_data[e][:,k].flatten(), z_score_data[ee][:,k].flatten())[0]
            #     norm_dist.append(util.sbd_to_norm(dist/len(clusters[j])))
            std = np.sqrt(np.mean(np.array(norm_dist)**2))
            if std == 0:
                std = util.sbd_to_norm(0.2)/np.sqrt(len(clusters[j]))
            cluster_distribution.append(norm(0, std))
        # zero_prob = 0.97
        print('out')
        pickle.dump({'cluster_distribution': cluster_distribution, **pattern_lib[k]},
                    open( path + r'\kpi_' + str(k), 'wb'))


#计算每个特征的信息熵
def feature_entropy(samples):
    samples = np.array(samples)
    entropys = []
    for i in range(samples.shape[1]):
        feature = samples[:, i]
        counts = np.bincount(feature)
        entropys.append(util.calculate_entropy(counts))
    return np.array(entropys)

#基于互信息的特征选择
def feature_selection(data, kpi_list):
    threshold = 0.1
    samples = []
    labels = []
    for t in range(len(data.anomaly_type)):
        samples.extend(data.pattern_feature[t])
        labels.extend([t] * len(data.pattern_feature[t]))

    # 计算互信息
    mi = mutual_info_classif(samples, labels, discrete_features=True)
    # # 选择k个最好的特征
    # mi_selector = SelectKBest(mutual_info_classif, k=4)
    # X_mi = mi_selector.fit_transform(samples, labels)
    # print(mi)
    entropy = feature_entropy(samples)
    label_entropy = util.calculate_entropy(np.bincount(labels))
    # print(entropy)
    # nmi = mi / (entropy + label_entropy) * 2
    # print(nmi)
    sqrt = np.sqrt(entropy * label_entropy)
    sqrt[sqrt == 0] = 1
    nmi = mi / sqrt
    print(nmi)
    print(np.array(kpi_list)[nmi > threshold])
    # from sklearn.ensemble import RandomForestClassifier
    #
    # # 训练随机森林模型
    # model = RandomForestClassifier()
    # model.fit(samples, labels)
    #
    # # 获取特征重要性
    # importances = model.feature_importances_
    # print(importances)
    feature_select = nmi > threshold
    kpi_select = np.array(kpi_list)[feature_select]
    return kpi_select




# 选择的KPI在所有KPI中的索引
# kpi_list = [1, 3, 4, 7, 8, 12, 13, 14, 15, 23, 30, 33, 69, 84, 105, 107, 109]
kpi_list = [1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 29, 30, 32, 33, 69, 84, 103, 104, 105, 107, 109, 110]
datasets_path = [r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\32_128\single\dataset',]
                 # r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\32_256\single\dataset',
                 # r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\64_128\single\dataset',
                 # r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\64_256\single\dataset']
datasets = []
# data = dataset(r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\64_256\single')
# data.load_data()
# data.preprocess(kpi_list)

# 把处理好的dataset存下来方便下次直接使用
# pickle.dump(data, open(r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\64_256\single\dataset', 'wb'))
# data = pickle.load(open(r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\64_256\single\dataset', 'rb'))

for path in datasets_path:
    datasets.append(pickle.load(open(path, 'rb')))

multidata = multidataset(datasets)
multidata.load()

#按比例划分训练和测试集
# train, test = data.split(0.2)
# pickle.dump(train, open(r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\32_128\single\train_20', 'wb'))
# pickle.dump(test, open(r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\32_128\single\test_20', 'wb'))
# train = pickle.load(open(r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\32_128\single\train_20', 'rb'))
# test = pickle.load(open(r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\32_128\single\test_20', 'rb'))



lib_path = r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\kpi_cluster'

# 对每种异常类型下的每个kpi进行基于异常值分布的聚类分析
# for k in kpi_list:
#     cluster_per_kpi(multidata, k, lib_path)
#
# analyze_clusters(multidata, kpi_list, lib_path)

# discretize(multidata, kpi_list, lib_path)
# kpi_select = feature_selection(multidata, kpi_list)
#
#
# model = pattern_model(multidata, lib_path, kpi_select)
# model.load()
# model.discretize()
# model.learn_bayesian()
# pickle.dump(model,open(r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\model', 'wb'))
# model = pickle.load(open(r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\model', 'rb'))
#
#
# total = {}
# for c in range(0, 4):
#     correct = 0
#     preds = [0]*model.fault_num
#     for i in range(len(multidata.aligned_data[c])):
#         ret = model.confidence(multidata.aligned_data[c][i],multidata.datasets[multidata.index[c][i]].dist)
#         # ret = pl.confidence(data.aligned_data[c][i])
#         print(ret.tolist())
#         pred = np.argmax(ret)
#         preds[pred] += 1
#         print(preds)
#         # print(ret)
#         if pred == c:
#             correct += 1
#     correct /= len(multidata.aligned_data[c])
#     total[c] = correct
#     print('fault'+str(c) + ': ' + str(correct))
#
# print(total)