# Assignment 5: K-Nearest Neighbours (KNN) Classification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.20, random_state=42, stratify=data.target
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k_values = [1, 3, 5, 7, 9]
scores = []

for k in k_values:
    model = KNeighborsClassifier(k).fit(X_train, y_train)
    scores.append(accuracy_score(y_test, model.predict(X_test)))

best_k = k_values[int(np.argmax(scores))]
model = KNeighborsClassifier(best_k).fit(X_train, y_train)
pred = model.predict(X_test)

print("Best K:", best_k)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

plt.plot(k_values, scores, marker="o")
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy for Different K Values")
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.show()
