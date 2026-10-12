# Assignment 6: Linear Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = load_diabetes()
X = data.data[:, [2]]     # BMI
Y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42
)

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)

print("MAE:", round(mean_absolute_error(y_test, pred), 4))
print("MSE:", round(mean_squared_error(y_test, pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 4))
print("R2:", round(r2_score(y_test, pred), 4))

plt.scatter(y_test, pred, s=35)
low = min(y_test.min(), pred.min())
high = max(y_test.max(), pred.max())
plt.plot([low, high], [low, high], linestyle="--")
plt.xlabel("Actual Target")
plt.ylabel("Predicted Target")
plt.title("Actual vs Predicted - Linear Regression")
plt.tight_layout()
plt.show()
