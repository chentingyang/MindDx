import pickle

from matplotlib import style, pyplot as plt
from scipy.stats import norm
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import networkx as nx
import diagnosis
import hierarchicalcluster
import sbd
import util


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



def triple_draw_normal(data, metric, index):
    draw_normal(data[index * 3], metric, 'b')
    draw_normal(data[index * 3 + 1], metric, 'g')
    draw_normal(data[index * 3 + 2], metric, 'r')


# 选择的KPI在所有KPI中的索引
# kpi_list = [1, 3, 4, 7, 8, 12, 13, 14, 15, 23, 30, 33, 69, 84, 105, 107, 109]
kpi_titile = ["Time","usr","sys","idl","wai","stl","read","writ","recv","send","in","out","used","free","buff","cach",
              "int","csw","run","blk","new","1m","5m","15m","used","free","343","344","416","read","writ","#aio",
              "files","inodes","msg","sem","shm","pos","lck","rea","wri","raw","tot","tcp","udp","raw","frg","lis",
              "act","syn","tim","clo","lis","act","dgm","str","lis","act","majpf","minpf","alloc","free","steal",
              "scanK","scanD","pgoru","astll","d32F","d32H","normF","normH","Conn","%Con","Act","LongQ","LongX",
              "Idl","LIdl","LWait","SQLs1","SQLs3","SQLs5","Xact1","Xact3","Locks","shared_buffers","work_mem",
              "bgwriter_delay","max_connections","autovacuum_work_mem","temp_buffers","autovacuum_max_workers",
              "maintenance_work_mem","checkpoint_timeout","max_wal_size","checkpoint_completion_target",
              "wal_keep_segments","wal_segment_size","clean","back","alloc","heapr","heaph","ratio","size","grow",
              "insert","update","delete","comm","roll"]
kpi_list = [1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 29, 30, 32, 33, 69, 84, 103, 104, 105, 107, 109, 110]
# data = dataset(r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\32_256\single')
# data.load_data()
# data.align_data()

# 把处理好的dataset存下来方便下次直接使用
# pickle.dump(data, open(r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\32_256\single\dataset', 'wb'))
data = pickle.load(open(r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\32_128\single\dataset', 'rb'))

#按比例划分训练和测试集
# train, test = data.split(0.2)
# pickle.dump(train, open(r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\32_128\single\train_20', 'wb'))
# pickle.dump(test, open(r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\32_128\single\test_20', 'wb'))
train = pickle.load(open(r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\32_128\single\train_20', 'rb'))
test = pickle.load(open(r'C:\Users\闫凌森\Desktop\diagnosis\DBPA_dataset-main\32_128\single\test_20', 'rb'))


# # 计算无异常情况下每个kpi的median和mad
# median = {}
# mad = {}
# for k in kpi_list:
#     ts = []
#     for t in data.normal_data:
#         # ts.append((t[:, kpi] - 1638.4)/1638.4)
#         ts.append(t[:, k])
#     median[k], mad[k] = util.calculate_median_mad(ts)
#     if mad[k] == 0:
#         mad[k] = 1
#
# variance = {}
# dist = {}
# for k in kpi_list:
#     variance[k] = []
#     for t in data.normal_data:
#         # ts.append((t[:, kpi] - 1638.4)/1638.4)
#         variance[k].append(np.var(t[:, k]))
#     # 拟合分布
#     # 使用scipy.stats.norm.fit来计算最适合数据的正态分布参数(均值，标准差)
#     params = norm.fit(variance[k])
#     # 创建一个正态分布的对象
#     dist[k] = norm(*params)
#     # 打印参数
#     print(f'均值: {params[0]}')
#     print(f'标准差: {params[1]}')

# f = Fitter(variance[1], distributions=['norm', 'gamma', 'rayleigh', 'uniform'], bins='auto', timeout=100)  # 创建Fitter类
# f.fit()  # 调用fit函数拟合分布
#
# print(f.summary())
# print(f.fitted_param)

# def correlation(data):
def draw_correlation(data):
    cors = []
    for i in range(len(data)):
        cor = np.corrcoef(np.array(data[i][:, kpi_list].tolist()), rowvar=False)
        cors.append(cor)
    cors = np.array(cors)
    median = np.median(cors, axis=0)
    mad = np.median(np.absolute(cors - median), axis=0)

    # 创建一个无向图
    G = nx.Graph()

    # 添加节点
    G.add_nodes_from(kpi_list)
    for i in range(len(mad)):
        for j in range(i + 1, len(mad[i])):
            if mad[i][j] <= 0.1 and median[i][j] >= 0.7:
                G.add_edge(kpi_list[i], kpi_list[j], weight=np.round(median[i][j],2))

    labels = {n: kpi_titile[n] for n in kpi_list}
    edges_labels = nx.get_edge_attributes(G, 'weight')

    # 设置布局
    pos = nx.spring_layout(G, k=1.5)

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos, labels, font_size=14, font_color='black')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edges_labels)
    plt.show()


draw_correlation(data.normal_data)
for f in range(9):
    draw_correlation(data.aligned_data[f])

# kpi = 84
# for j in range(0, 9):
#     for i in range(1, 3):
#         triple_draw_normal(data.raw_data[j], kpi, i)
#     plt.title("fault" + str(j) + "_KPI" + str(kpi))
#     plt.show()
# for i in range(0, 3):
#     triple_draw_normal(data.normal_data, kpi, i)
# plt.title("normal_KPI" + str(kpi))
# plt.show()






