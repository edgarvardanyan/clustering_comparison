from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import \
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 10)

data = make_blobs(
    n_samples=200,
    n_features=2,
    centers=8
)

plt.scatter(data[0][:, 0], data[0][:, 1])
plt.show()

models = []

for i in range(2, 16):
    models.append((KMeans(n_clusters=i), f"KMeans with k={i}"))

results = defaultdict(list)

for model, description in models:
    predicted = model.fit_predict(data[0])
    results['Model'].append(description)
    results['silhouette_score'].append(
        silhouette_score(data[0], predicted, metric='euclidean'))
    results['calinski_harabasz_score'].append(
        calinski_harabasz_score(data[0], predicted))
    results['davies_bouldin_score'].append(
        davies_bouldin_score(data[0], predicted))

results = pd.DataFrame(results)
print(results)
