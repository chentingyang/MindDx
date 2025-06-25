import pickle


import numpy as np
import pandas as pd
from opt_einsum import contract
from pgmpy.estimators import PC, MaximumLikelihoodEstimator, MmhcEstimator, BDeuScore, HillClimbSearch, BicScore, \
    BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

import sbd
import util

#样本形状离散化表示
def discretize_value(data, kpi_list):
    threshold = 0.2
    data.value_feature = []
    for fault in range(len(data.anomaly_type)):
        data.value_feature.append([])
        for i in range(len(data.aligned_data[fault])):
            data.value_feature[fault].append([])
    for fault in range(len(data.anomaly_type)):
        for i in range(len(data.aligned_data[fault])):
            for k in kpi_list:
                t = (data.aligned_data[fault][i][:, k] - data.datasets[data.index[fault][i]].median[k]) / data.datasets[data.index[fault][i]].mad[k]
                cnt_low = 0
                cnt_up = 0
                for e in t:
                    cdf = util.kpi_cdf3(e)
                    if cdf < 0.03:
                        cnt_low += 1
                    elif cdf > 0.97:
                        cnt_up += 1
                if cnt_up > threshold * len(t):
                    data.value_feature[fault][i].append(2)
                elif cnt_low > threshold * len(t):
                    data.value_feature[fault][i].append(0)
                else:
                    data.value_feature[fault][i].append(1)


#样本形状离散化表示
def discretize(data, kpi_list, path):
    data.pattern_feature = []
    pattern_lib = {}
    for k in kpi_list:
        pattern_lib[k] = pickle.load(open(path + r'\kpi_' + str(k), 'rb'))
    feature = []
    for t in range(len(data.anomaly_type)):
        for i in range(len(data.z_score_data[t])):
            feature.append([])
    for i in range(len(kpi_list)):
        k = kpi_list[i]
        clusters = pattern_lib[k]['clusters']
        for j in range(len(clusters)):
            for e in clusters[j]:
                feature[e].append(j + 1)
        for f in feature:
            if len(f) <= i:
                f.append(0)

    for t in range(len(data.anomaly_type)):
        data.pattern_feature.append(feature[:len(data.z_score_data[t])])
        feature = feature[len(data.z_score_data[t]):]


#用于贝叶斯推理
class ShapeBayesianInference(VariableElimination):

    def infer(self, shape_probability:dict):
        var_int_map = {var: i for i, var in enumerate(self.model.nodes())}
        einsum_expr = []
        factors = self.model.cpds
        for index, phi in enumerate(factors):
            einsum_expr.append(phi.values)
            einsum_expr.append(
                [
                    var_int_map[var]
                    for var in phi.variables
                ]
            )

        for node, prob in shape_probability.items():
            einsum_expr.append(np.array(prob))
            einsum_expr.append(
                [
                    var_int_map[node]
                ]
            )
        result_values = contract(
            *einsum_expr, [], optimize="greedy"
        )
        return result_values


# 形状模式库
class pattern_model:

    def __init__(self, dataset, path, kpi_list):
        self.path = path
        self.fault_num = len(dataset.anomaly_type)
        # self.kpi_list = [1, 3, 4, 7, 8, 12, 13, 14, 15, 23, 30, 33, 69, 84, 105, 107, 109]
        self.kpi_list = kpi_list
        # self.kpi_list = [i for i in range(1, 85)] + [i for i in range(98, 111)]
        self.pattern_lib = {}
        self.dataset = dataset
        self.bayesian_net = []
        self.bayesian_inference = []
        self.zero_prob = 0.97

        #计算异常类型的先验，可根据专家知识指定
        # prior = []
        # for f in range(self.fault_num):
        #     prior.append(len(dataset.z_score_data[f]))
        # self.prior = np.array(prior) / np.sum(prior)
        # self.prior = [1] * self.fault_num


    def load(self):
        for k in self.kpi_list:
            clusters = pickle.load(open(self.path + r'\kpi_' + str(k), 'rb'))
            self.pattern_lib[k] = clusters

    #将样本离散化表示
    def discretize(self):
        discretize(self.dataset, self.kpi_list, self.path)

    #学习贝叶斯网络
    def learn_bayesian(self):
        for f in range(self.fault_num):
            samples = pd.DataFrame(self.dataset.pattern_feature[f])
            if f != 70: ####################
                mmhc = HillClimbSearch(samples)
                best_model = mmhc.estimate(scoring_method=BDeuScore(samples))
            else:
                pc = PC(samples)
                best_model = pc.estimate(ci_test='chi_square', return_type='dag')

            # 输出贝叶斯网络的边
            print(best_model.edges())

            # 创建贝叶斯网络模型
            model = BayesianNetwork()
            model.add_nodes_from([i for i in range(len(self.kpi_list))])
            model.add_edges_from(best_model.edges())

            # 使用最大似然估计进行参数估计
            model.fit(samples, estimator=MaximumLikelihoodEstimator)

            # # 打印学习到的参数
            # print("\nLearned CPDs:")
            # for cpd in model.get_cpds():
            #     print(cpd)

            self.bayesian_net.append(model)
            self.bayesian_inference.append(ShapeBayesianInference(model))

    #计算样本在每一个聚类分布中的概率
    def get_shape_probs(self, sample, var_distribution):
        shape_prob = {}
        for k in range(len(self.kpi_list)):
            kpi = self.kpi_list[k]
            clusters = self.pattern_lib[kpi]['clusters']
            cluster_distributions = self.pattern_lib[kpi]['cluster_distribution']
            prob = []
            x = sample[:, kpi].flatten()
            z_score_x = util.z_score_normalization(x)
            if not var_distribution[kpi].cdf(np.var(x)) > 0.97:
                prob.append(self.zero_prob)
                for i in range(len(clusters)):
                    prob.append(1 - self.zero_prob)
            else:
                prob.append(1 - self.zero_prob)
                for i in range(len(clusters)):
                    dist = 0
                    for e in clusters[i]:
                        dist += sbd.sbd_distance(z_score_x, self.dataset.full_z_score_data[e][:, kpi].flatten())[0]
                    dist = dist / len(clusters[i])
                    prob.append(util.sbd_norm_pdf(cluster_distributions[i], dist))
            shape_prob[kpi] = np.array(prob)
        shape_probs = []
        for f in range(self.fault_num):
            t = {}
            states = self.bayesian_net[f].states
            for node, value in states.items():
                t[node] = shape_prob[self.kpi_list[node]][value]
            shape_probs.append(t)
        return shape_probs

    #计算分类概率
    def confidence(self, sample, var_distribution):
        conf = []
        shape_probs = self.get_shape_probs(sample, var_distribution)
        for f in range(self.fault_num):
            conf.append(self.bayesian_inference[f].infer(shape_probs[f]))
        # conf = np.array(conf) * self.prior
        # conf = conf / np.sum(conf)
        return conf




# 值分布模式库
class value_model:

    def __init__(self, dataset, kpi_list):
        self.fault_num = len(dataset.anomaly_type)
        # self.kpi_list = [1, 3, 4, 7, 8, 12, 13, 14, 15, 23, 30, 33, 69, 84, 105, 107, 109]
        self.kpi_list = kpi_list
        # self.kpi_list = [i for i in range(1, 85)] + [i for i in range(98, 111)]
        self.dataset = dataset
        self.bayesian_net = []

        # 计算异常类型的先验，可根据专家知识指定
        # prior = []
        # for f in range(self.fault_num):
        #     prior.append(len(dataset.z_score_data[f]))
        # self.prior = np.array(prior) / np.sum(prior)
        # self.prior = [1] * self.fault_num


    # 将样本离散化表示
    def discretize(self):
        discretize_value(self.dataset, self.kpi_list)

    # 学习贝叶斯网络
    def learn_bayesian(self):
        for f in range(self.fault_num):
            samples = pd.DataFrame(self.dataset.value_feature[f])
            if f != 77:  ####################
                mmhc = HillClimbSearch(samples)
                best_model = mmhc.estimate(scoring_method=BicScore(samples))
            else:
                pc = PC(samples)
                best_model = pc.estimate(ci_test='chi_square', return_type='dag')

            # 输出贝叶斯网络的边
            print(best_model.edges())

            # 创建贝叶斯网络模型
            model = BayesianNetwork()
            model.add_nodes_from([i for i in range(len(self.kpi_list))])
            model.add_edges_from(best_model.edges())

            # 使用最大似然估计进行参数估计
            model.fit(samples,state_names={i:[0,1,2] for i in range(len(self.kpi_list))}, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=1)

            # # 打印学习到的参数
            # print("\nLearned CPDs:")
            # for cpd in model.get_cpds():
            #     print(cpd)

            self.bayesian_net.append(model)


    # 计算分类概率
    def confidence(self, sample, median, mad):
        threshold = 0.2
        conf = []
        feature = []
        for k in self.kpi_list:
            t = (sample[:, k] - median[k]) / mad[k]
            cnt_low = 0
            cnt_up = 0
            for e in t:
                cdf = util.kpi_cdf3(e)
                if cdf < 0.03:
                    cnt_low += 1
                elif cdf > 0.97:
                    cnt_up += 1
            if cnt_up > threshold * len(t):
                feature.append(2)
            elif cnt_low > threshold * len(t):
                feature.append(0)
            else:
                feature.append(1)
        state = {i: feature[i] for i in range(len(feature))}
        for f in range(self.fault_num):
            conf.append(self.bayesian_net[f].get_state_probability(state))
        # conf = np.array(conf) * self.prior
        # conf = conf / np.sum(conf)
        return conf

class fusion_model:
    def __init__(self, pm, vm):
        self.pm = pm
        self.vm = vm
        self.fault_num = self.pm.fault_num
        self.prior = [1] * self.fault_num

    def confidence(self, sample, var_distribution, median, mad):
        pattern_confidence = self.pm.confidence(sample, var_distribution)
        value_confidence = self.vm.confidence(sample, median, mad)
        confidence = np.array(pattern_confidence) * np.array(value_confidence)
        confidence = confidence * self.prior
        confidence = confidence /np.sum(confidence)
        return confidence