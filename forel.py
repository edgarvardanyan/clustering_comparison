import numpy as np
from random import choice, shuffle
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_blobs


class ForelClustering:

    def __init__(self, r: float):
        self.r = r

    def fit_predict(self, data):
        n_samples = data.shape[0]
        clusters = []
        labels = [None for _ in range(n_samples)]
        unlabeled_elements = list(range(n_samples))
        next_label = 0
        while unlabeled_elements:
            next_cluster = []
            elem = choice(unlabeled_elements)
            center = data[elem]
            next_cluster.append(elem)
            unlabeled_elements.remove(elem)
            while True:
                next_elem = None
                shuffle(unlabeled_elements)
                for elem in unlabeled_elements:
                    if self.distance(data[elem], center) <= self.r:
                        next_elem = elem
                        break
                if next_elem is not None:
                    unlabeled_elements.remove(next_elem)
                    # print(len(unlabeled_elements))
                    plt.scatter(data[unlabeled_elements][:, 0], data[unlabeled_elements][:, 1], c='black')
                    plt.scatter(data[next_cluster][:, 0], data[next_cluster][:, 1], c='green')
                    plt.scatter(data[[next_elem]][:, 0], data[[next_elem]][:, 1], c='yellow')
                    for cluster in clusters:
                        plt.scatter(data[cluster][:, 0], data[cluster][:, 1], c='red')
                    plt.show()
                    time.sleep(0.3)
                    next_cluster.append(elem)
                    center = np.mean(data[next_cluster], axis=0)
                else:
                    clusters.append(next_cluster)
                    for elem in next_cluster:
                        labels[elem] = next_label
                    next_label += 1
                    break
        return np.array(labels)

    @staticmethod
    def distance(p1, p2):
        return np.sum((p1 - p2)**2) ** 0.5


if __name__ == '__main__':
    data = make_blobs(
        n_samples=200,
        n_features=2,
        centers=16
    )
    r = 3
    model = ForelClustering(r=r)
    model.fit_predict(data[0])
