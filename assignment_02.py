# Assignment 2: K-Means Clustering Using Unsupervised Learning
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = load_iris()
X = StandardScaler().fit_transform(data.data[:, :2])

model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(X)

print("Silhouette Score:", round(silhouette_score(X, labels), 4))
print("Cluster Sizes:", np.bincount(labels))

for k in range(3):
    p = labels == k
    plt.scatter(X[p, 0], X[p, 1], s=30, label=f"Cluster {k+1}")

plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    marker="X", s=100, c="black", label="Centroids"
)
plt.xlabel("Standardized Sepal Length")
plt.ylabel("Standardized Sepal Width")
plt.title("K-Means Clustering on Iris")
plt.legend()
plt.tight_layout()
plt.show()
