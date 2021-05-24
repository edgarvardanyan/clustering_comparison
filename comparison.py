import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import \
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

from forel import ForelClustering

pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 10)

# data = make_blobs(
#     n_samples=1000,
#     n_features=2,
#     centers=10,
#     random_state=10,
# )
data = np.array(pd.read_csv('/home/edgar/rau/clustering/data.csv'))

# plt.scatter(data[:, 0], data[:, 1])
# plt.show()

models = []

# for i in range(2, 16):
#     models.append((KMeans(n_clusters=i), f"KMeans with k={i}"))

for i in np.linspace(20, 400, 20):
    models.append((ForelClustering(r=i), f"Forel with r={i}"))

results = defaultdict(list)

for model, description in tqdm(models):
    predicted = model.fit_predict(data)
    results['Model'].append(description)
    results['silhouette_score'].append(
        silhouette_score(data, predicted, metric='euclidean'))
    results['calinski_harabasz_score'].append(
        calinski_harabasz_score(data, predicted))
    results['davies_bouldin_score'].append(
        davies_bouldin_score(data, predicted))

    df = pd.DataFrame(results)
    print(df)
    print("\n"*10)
