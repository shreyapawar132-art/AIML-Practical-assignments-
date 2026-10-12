# Assignment 8: Regularization to Avoid Overfitting
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

rng = np.random.RandomState(42)
x = np.linspace(-3, 3, 120)
y = 2*x**2 + 1.5*x + rng.normal(0, 2, 120)

poly = PolynomialFeatures(10, include_bias=False)
X = poly.fit_transform(x.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1),
    "Lasso": Lasso(alpha=0.05, max_iter=20000)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(name, "R2:", round(r2, 4), "RMSE:", round(rmse, 4))

# Visual comparison of predictions
xx = np.linspace(-3, 3, 200)
XX = poly.transform(xx.reshape(-1, 1))

plt.scatter(x, y, s=20, label="Data")
for name, model in models.items():
    plt.plot(xx, model.predict(XX), label=name)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Regularization Comparison")
plt.legend()
plt.tight_layout()
plt.show()
