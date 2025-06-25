import multiprocessing
import pickle
import threading

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt

from matplotlib import style

import numpy as np
from scipy.stats import norm

import diagnosis
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


def calulate_s(ts):
    sum = 0
    for i in ts:
        sum += np.fabs(i)
    sum = np.fabs(sum)
    sum = sum/np.sqrt(len(ts))
    return sum


# 绘制kpi曲线
def triple_draw_normal(data, metric, index):
    draw_normal(data[index * 3], metric, 'b')
    draw_normal(data[index * 3 + 1], metric, 'g')
    draw_normal(data[index * 3 + 2], metric, 'r')


# 对data数据集中的fault类型下的metricKPI进行基于形状的聚类分析并将结果保存到文件
def cluster_per_fault(data, fault, metric):
    ts = []
    index = []
    zero_cluster = 0
    for i in range(len(data.z_score_data[fault])):
        if data.datasets[data.index[fault][i]].dist[metric].cdf(np.var(data.aligned_data[fault][i][:, metric])) > 0.97:
            index.append(len(ts) + zero_cluster)
            ts.append(data.z_score_data[fault][i][:, metric].flatten())
        else:
            zero_cluster += 1
    ret = hierarchicalcluster.hierarchical_clustering(ts, 1, 0.4)
    print("Total:" + str(len(ts) + zero_cluster))
    ts = np.array(ts).transpose()
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
                open(r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\kpi_pattern30\fault' + str(
                    fault + 1) + r'\kpi_' + str(metric), 'wb'))


# 对data数据集中的fault类型下的metricKPI进行基于值分布的聚类分析并将结果保存到文件
def cluster_distribution_per_fault(data, fault, metric):
    raw = []
    ts = []
    index = []
    zero_cluster = 0

    for i in range(len(data.aligned_data[fault])):
        t = (data.aligned_data[fault][i][:, metric] - data.datasets[data.index[fault][i]].median[metric]) / data.datasets[data.index[fault][i]].mad[metric]
        cnt_low = 0
        cnt_up = 0
        for e in t:
            cdf = util.kpi_cdf3(e)
            if cdf < 0.03:
                cnt_low += 1
            elif cdf > 0.97:
                cnt_up += 1
        if cnt_up == 0 and cnt_low == 0:
            zero_cluster += 1
        else:
            index.append(len(ts) + zero_cluster)
            ts.append([cnt_low, len(t)-cnt_low-cnt_up, cnt_up])
        raw. append([cnt_low, len(t)-cnt_low-cnt_up, cnt_up])
    ret = hierarchicalcluster.hierarchical_clustering(ts, 1, 0.3, dist='wasserstein')
    print("Total:" + str(len(ts) + zero_cluster))
    # for i in range(len(ts)):
    #     plt.plot(ts[i])
    # plt.show()
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
    #             plt.plot(raw[j])
    #         plt.show()
    print('cluster  zero: ' + str(zero_cluster))
    pickle.dump({'clusters': clusters, 'zero': zero_cluster, 'distributions': raw},
                open(r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\kpi_distribution30\fault' + str(
                    fault + 1) + r'\kpi_' + str(metric), 'wb'))


# 暂时无用
def cluster_per_kpi(data, metric):
    ts = []
    num = []
    for i in range(9):
        num.append(min(30, len(data.z_score_data[i])))
    for j in range(9):
        for i in range(num[j]):
            ts.append(data.z_score_data[j][i][:, metric].flatten())

    ret = hierarchicalcluster.hierarchical_clustering(ts, 1, 100)

    ts = np.array(ts).transpose()

    s = 0
    for j in range(9):
        for i in range(s, s + 30):
            plt.plot(ts[:, i])
        plt.show()
        s += num[j]

    # ret = dbscan.dbscan(ts, 0.15, 1)

    s = 0
    for j in range(9):
        print(ret[s:s + num[j]])
        s += num[j]

    cluster = 0
    for i in ret:
        if i is not None and i > cluster:
            cluster = i
    clusters = [[] for _ in range(cluster + 1)]
    for i in range(len(ret)):
        if ret[i] is not None:
            clusters[ret[i]].append(i)
    for i in range(len(clusters)):
        if len(clusters[i]) > 5:
            for j in clusters[i]:
                plt.plot(ts[:, j])
            plt.show()


# 选择的KPI在所有KPI中的索引
# kpi_list = [1, 3, 4, 7, 8, 12, 13, 14, 15, 23, 30, 33, 69, 84, 105, 107, 109]
kpi_list = [1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 29, 30, 32, 33, 69, 84, 103, 104, 105, 107, 109, 110]
datasets_path = [r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\32_128\single\dataset',
                 r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\32_256\single\dataset',
                 r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\64_128\single\dataset',
                 r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\64_256\single\dataset']
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

style.use('ggplot')
# for j in [105, 33]:
#     for i in range(15, 18):
#         triple_draw_normal(data.z_score_data[4], j, i)
#     plt.show()
# kpi = 1
# for j in range(1, 2):
#     for i in range(0, 1):
#         triple_draw_normal(data.z_score_data[j], kpi, i)
#     plt.title("fault" + str(j) + "_KPI" + str(kpi))
#     plt.show()
# for i in range(0, 10):
#     triple_draw_normal(data.normal_data, kpi, i)
# plt.title("normal_KPI" + str(kpi))
# plt.show()



# p = []
# # 对每种异常类型下的每个kpi进行基于形状的聚类分析
# for i in [0]:
#     p.append(threading.Thread(target=cluster_fault, args=(mutidata, kpi_list, i)))
#     p[-1].start()

# for i in range(0, 9):
#     for k in kpi_list:
#         cluster_per_fault(mutidata, i, k)



# 对每种异常类型下的每个kpi进行基于异常值分布的聚类分析
# for f in range(0, 9):
#     for k in kpi_list:
#         cluster_distribution_per_fault(multidata, f, k)


# ts = []
# distribution = []
# for t in data.aligned_data[7]:
#     # ts.append((t[:, kpi] - 1638.4)/1638.4)
#     ts.append(t[:, kpi])
# for i in range(len(ts)):
#     ts[i] = (ts[i] - median) / mad
# for i in range(len(ts)):
#     cnt_low = 0
#     cnt_up = 0
#     t = ts[i]
#     # plt.plot(t)
#     # plt.show()
#     for e in t:
#         cdf = util.kpi_cdf3(e)
#         if cdf < 0.1:
#             cnt_low += 1
#         elif cdf > 0.9:
#             cnt_up += 1
#         # print(cdf)
#     print(cnt_low / len(t), cnt_up / len(t))
#     distribution.append([cnt_low, len(t) - cnt_low - cnt_up, cnt_up])
# for i in range(20):
#     for j in range(20):
#         print(util.wasserstein_dist(distribution[i], distribution[j]))


# cnt = 0
# for i in range(2*3, 80*3):
#     # dfgls = DFGLS(data.aligned_data[1][i][:, kpi].flatten().astype(np.float64), trend='c')
#     # adf = ADF(data.aligned_data[1][i][:, kpi].flatten().astype(np.float64), trend='c')
#     # kpss = KPSS(data.aligned_data[4][i][:, kpi].flatten().astype(np.float64), trend='c')
#     pp = PhillipsPerron(data.aligned_data[1][i][:, kpi].flatten().astype(np.float64), trend='c')
#     print(1- pp.pvalue)
#     if pp.pvalue < 0.05:
#         cnt += 1
#         plt.plot(data.aligned_data[1][i][:, kpi].flatten().astype(np.float64))
#         plt.show()
# print(cnt)
    # plt.plot(data.aligned_data[4][i][:, kpi].flatten().astype(np.float64))
    # plt.show()

    # 设置横坐标刻度
    # plt.xticks(range(0, 40))
    # plt.xlim(0, 20)
    # 增加竖直分割线
    # plt.axvline(x=6, color='m', linestyle='--')
    # plt.axvline(x=20, color='m', linestyle='--')

# for i in range(0, 10):
#     plt.plot(data.raw_data[2][i][:, 4].flatten())
#     plt.show()
#     util.check_shift(data.raw_data[2][i][0:20, 4].flatten().astype(float))

#
# x = []
# y = []
# for i in range(25):
#     x.append(data.z_score_data[1][i][:, 12].flatten())
#     y.append(data.z_score_data[1][i][:, 12].flatten())
# dist = util.sbd_matrix(x, y)
# for row in dist:
#     for element in row:
#         print(element[0], end=" ")
#     print()

# print(sbd.sbd_distance([0,0,0],[0,0,0]))

# pl = diagnosis.pattern_lib(data, r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\32_128\single\kpi_pattern')
# pl.load()

# for c in range(2, 9):
#     correct = 0
#     preds = [0]*pl.fault_num
#     for i in range(len(data.z_score_data[c])):
#         ret = pl.confidence(data.z_score_data[c][i])
#         pred = np.argmin(ret)
#         preds[pred] += 1
#         print(preds)
#         if pred == c:
#             correct += 1
#     correct /= len(data.z_score_data[c])
#     print('fault'+str(c) + ': ' + str(correct))


# pl = diagnosis.distribution_lib(data, r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\32_128\single\kpi_distribution', median, mad)
# pl.load()

# 创建一个联合诊断器
pl = diagnosis.diagnotor(multidata, r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\kpi_pattern30',
                           r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\kpi_distribution30')
#
# data = test

# 将数据集中的所有样本进行诊断，测试结果
for c in range(0, 9):
    correct = 0
    preds = [0]*pl.fault_num
    for i in range(len(multidata.aligned_data[c])):
        ret = pl.confidence(multidata.aligned_data[c][i], multidata.datasets[multidata.index[c][i]].median,multidata.datasets[multidata.index[c][i]].mad,multidata.datasets[multidata.index[c][i]].dist)
        # ret = pl.confidence(data.aligned_data[c][i])
        pred = np.argmin(ret)
        preds[pred] += 1
        print(preds)
        # print(ret)
        if pred == c:
            correct += 1
    correct /= len(multidata.aligned_data[c])
    print('fault'+str(c) + ': ' + str(correct))



# for i in range(5, 6):
#     # ret = pl.confidence(data.aligned_data[1][i], data.z_score_data[1][i])
#     ret = pl.confidence(data.aligned_data[1][i])
#     print(ret)

# a = []
# a.append(1)
# a.append(2)
# a.append(3)
#
# print(np.full_like(a, 1/len(a), dtype=float))
# print(a * np.full_like(a, 1/len(a), dtype=float))


