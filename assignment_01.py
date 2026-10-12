# Assignment 1: Association Between Dependent and Independent Variables
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

data = load_diabetes()
X = data.data[:, 2]       # BMI
Y = data.target           # Disease progression

r = np.corrcoef(X, Y)[0, 1]
slope, intercept = np.polyfit(X, Y, 1)

print("Correlation Coefficient:", round(r, 4))
print("Slope:", round(slope, 2))
print("Intercept:", round(intercept, 2))

plt.scatter(X, Y, s=35, label="Actual Data")
xx = np.linspace(X.min(), X.max(), 100)
plt.plot(xx, slope * xx + intercept, label="Trend Line")
plt.xlabel("BMI (standardized)")
plt.ylabel("Disease Progression")
plt.title("Association Between BMI and Disease Progression")
plt.legend()
plt.tight_layout()
plt.show()
