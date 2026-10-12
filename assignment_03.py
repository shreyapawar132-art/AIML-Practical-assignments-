# Assignment 3: Dimensionality Reduction Using PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_wine()
X = StandardScaler().fit_transform(data.data)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("PC1:", round(pca.explained_variance_ratio_[0] * 100, 2), "%")
print("PC2:", round(pca.explained_variance_ratio_[1] * 100, 2), "%")
print("Total:", round(pca.explained_variance_ratio_.sum() * 100, 2), "%")

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data.target, cmap="viridis", s=25)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Wine Dataset after PCA")
plt.tight_layout()
plt.show()
