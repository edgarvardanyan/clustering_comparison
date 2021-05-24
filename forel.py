import numpy as np
from random import choice, shuffle
import matplotlib.pyplot as plt

import pandas as pd
from itertools import cycle
from tqdm import tqdm


class ForelClustering:

    def __init__(self, r: float):
        self.r = r

    def fit_predict(self, data, plot=False):
        if plot:
            plt.scatter(data[:, 0], data[:, 1], s=1)
            plt.show()
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
                    next_cluster.append(elem)
                    center = np.mean(data[next_cluster], axis=0)
                else:
                    if len(next_cluster) >= 100:
                        clusters.append(next_cluster)
                    for elem in next_cluster:
                        labels[elem] = next_label
                    next_label += 1
                    break
        print(f"Found {len(clusters)} clusters")
        colors = cycle([
            "black",
            "silver",
            "gray",
            "maroon",
            "red",
            "purple",
            "fuchsia",
            "green",
            "lime",
            "olive",
            "yellow",
            "navy",
            "blue",
            "teal",
            "aqua"]
        )
        if plot:
            plt.scatter(data[unlabeled_elements][:, 0], data[unlabeled_elements][:, 1], c='black')
            for cluster, color in tqdm(zip(clusters, colors)):
                plt.scatter(data[cluster][:, 0], data[cluster][:, 1], c=color, s=1)
            plt.show()
        return np.array(labels)

    @staticmethod
    def distance(p1, p2):
        return np.sum((p1 - p2)**2) ** 0.5


if __name__ == '__main__':
    data = np.array(pd.read_csv('/home/edgar/rau/clustering/data.csv'))
    r = 150
    model = ForelClustering(r=r)
    model.fit_predict(data, plot=True)
