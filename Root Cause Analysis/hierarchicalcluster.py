import numpy as np

import sbd
import util


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def hierarchical_clustering(data, min_clusters=1, max_distance=np.inf, dist='sbd'):
    distances = np.zeros((len(data), len(data)))
    distances.fill(np.inf)
    np.fill_diagonal(distances, 0)

    for i in range(len(data)):
        for j in range(i+1, len(data)):
            if dist == 'sbd':
                distances[i, j] = sbd.sbd_distance(data[i].flatten(), data[j].flatten())[0]
            else:
                distances[i, j] = util.wasserstein_dist(data[i], data[j])
            # distances[i, j] = euclidean_distance(data[i], data[j])
            distances[j, i] = distances[i, j]

    clusters = [[i] for i in range(len(data))]
    while len(clusters) > min_clusters:
        min_distance = np.inf
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                cluster_distance = 0
                for k in clusters[i]:
                    for l in clusters[j]:
                        cluster_distance += distances[k, l]
                cluster_distance /= (len(clusters[i]) * len(clusters[j]))
                if cluster_distance < min_distance:
                    min_distance = cluster_distance
                    merge_indices = (i, j)
        if min_distance > max_distance:
            break
        # print(min_distance)
        i, j = merge_indices
        clusters[i].extend(clusters[j])
        del clusters[j]
    labels = [1] * len(data)
    for i in range(len(clusters)):
        for j in clusters[i]:
            labels[j] = i
    return labels

# 测试
# data = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])
# ret = hierarchical_clustering(data, 1)
# print(ret)
# for i, cluster in enumerate(clusters):
#     print(f"Cluster {i+1}: {cluster}")
