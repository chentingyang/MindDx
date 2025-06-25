import pandas as pd
from pgmpy.estimators import PC
from pgmpy.models import BayesianNetwork

# 准备数据
data = pd.read_csv(r'C:\Users\闫凌森\Desktop\test.csv')

# 使用PC算法进行结构学习
pc = PC(data)
model = pc.estimate(ci_test='g_sq')

# 输出贝叶斯网络的边
print(model.edges())
print(model)

# 可视化贝叶斯网络（需要pydot和matplotlib库）
import matplotlib.pyplot as plt
import networkx as nx

plt.figure(figsize=(8, 6))
nx.draw(model)
plt.show()
