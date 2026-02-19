# Polynomial Regression from Scratch

This document explains **Polynomial Regression** exactly the way it is done in practice — with **theory, math intuition, and line-by-line code explanation**.
You can directly push this document to GitHub as learning notes.

---

## 1. Why Polynomial Regression?

Linear Regression assumes a straight-line relationship:

[ y = w_0 + w_1 x ]

But many real-world relationships are **curved**:

* Delivery time vs distance
* Salary vs experience
* Growth vs time

Polynomial Regression solves this by **adding powers of x**, while still using Linear Regression internally.

---

## 2. Key Insight (MOST IMPORTANT)

> **Polynomial Regression is NOT a new algorithm**
> It is **Linear Regression applied to transformed (polynomial) features**.

So all concepts you already learned (MSE, Gradient Descent, Scaling) remain the same.

---

## 3. Dataset Creation (Simulating Real World)

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X = np.linspace(0, 10, 50)
y = 0.5 * X**2 + X + 3 + np.random.randn(50) * 4
```

### What each part means:

* `X = np.linspace(0, 10, 50)`
  Creates 50 evenly spaced input values between 0 and 10.

* `0.5 * X**2 + X + 3`
  Hidden **true relationship** (unknown to the model):
  [ y = 0.5x^2 + x + 3 ]

* `np.random.randn(50) * 4`
  Adds **noise** to simulate real-world randomness.

> In real ML problems, we NEVER know this true equation — this is only for simulation.

---

## 4. Visualizing the Data

```python
plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Raw Data with Hidden Nonlinear Pattern")
plt.show()
```

### Intuition:

* Data is curved
* Data is noisy
* A straight line will underfit

---

## 5. Train–Validation Split (Critical Step)

```python
indices = np.arange(len(X))
np.random.shuffle(indices)

train_size = int(0.7 * len(X))
train_idx = indices[:train_size]
val_idx = indices[train_size:]

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]
```

### Why this is required:

* Training error alone is misleading
* Validation error tells us **generalization ability**

---

## 6. Polynomial Feature Generator

```python
def polynomial_features(x, degree):
    X_poly = np.ones((len(x), degree + 1))
    for d in range(1, degree + 1):
        X_poly[:, d] = x ** d
    return X_poly
```

### What this does:

Converts:
[ x \rightarrow [1, x, x^2, x^3, ...] ]

### Math meaning:

For degree = 2:
[
X = \begin{bmatrix}
1 & x & x^2
\end{bmatrix}
]

---

## 7. Feature Scaling (Mandatory)

```python
def scale_features(X):
    mean = X[:, 1:].mean(axis=0)
    std = X[:, 1:].std(axis=0)
    X_scaled = X.copy()
    X_scaled[:, 1:] = (X[:, 1:] - mean) / std
    return X_scaled, mean, std
```

### Why scaling is required:

* Polynomial terms (x², x³) grow fast
* Without scaling → Gradient Descent diverges

### Math:

[
x_{scaled} = \frac{x - \mu}{\sigma}
]

---

## 8. Gradient Descent Training Function

```python
def train_gd(X, y, lr=0.05, epochs=3000):
    y = y.reshape(-1, 1)
    theta = np.zeros((X.shape[1], 1))
    n = len(y)

    for _ in range(epochs):
        y_pred = X @ theta
        error = y_pred - y
        grad = (2/n) * (X.T @ error)
        theta -= lr * grad

    return theta
```

### Math behind this:

* Prediction:
  [ \hat{y} = X\theta ]

* Loss (MSE):
  [ J(\theta) = \frac{1}{n} \sum (\hat{y} - y)^2 ]

* Gradient:
  [ \nabla J = \frac{2}{n} X^T (X\theta - y) ]

* Update:
  [ \theta := \theta - \alpha \nabla J ]

---

## 9. Mean Squared Error Function

```python
def mse(X, y, theta):
    y = y.reshape(-1, 1)
    preds = X @ theta
    return np.mean((preds - y) ** 2)
```

Used to compare **training vs validation performance**.

---

## 10. Trying Multiple Polynomial Degrees (PROFESSIONAL WAY)

```python
degrees = [1, 2, 3, 4, 6, 8]
results = []

for d in degrees:
    Xtr_poly = polynomial_features(X_train, d)
    Xval_poly = polynomial_features(X_val, d)

    Xtr_scaled, mean, std = scale_features(Xtr_poly)
    Xval_scaled = Xval_poly.copy()
    Xval_scaled[:, 1:] = (Xval_poly[:, 1:] - mean) / std

    theta = train_gd(Xtr_scaled, y_train)

    train_err = mse(Xtr_scaled, y_train, theta)
    val_err = mse(Xval_scaled, y_val, theta)

    results.append((d, train_err, val_err))
```

### What we are doing:

* Degree is a **hyperparameter**
* We try multiple degrees
* Data decides the best model

---

## 11. Interpreting Results

Typical pattern:

* Degree 1 → Underfitting (high train & val error)
* Degree 2 → Best generalization (lowest val error)
* High degree → Overfitting (low train, high val)

> **We select the degree with minimum validation error.**

---

## 12. Core Intuition (FINAL TAKEAWAY)

* Polynomial Regression = Linear Regression + Feature Engineering
* Degree controls **model flexibility**, not feature count
* We NEVER guess the true equation
* Validation error chooses the degree
* Scaling is mandatory

---

## 13. One-Line Summary

> **Polynomial Regression is hypothesis testing with controlled complexity, guided by validation data.**

---

### You can now safely push this document to GitHub as:

```
polynomial-regression-from-scratch.md
```

This reflects **real ML understanding**, not cookbook learning.
