import numpy as np

np.random.seed(0)

# -------------------------
# 1️⃣ Create Data
# -------------------------
n = 100

x1 = np.random.uniform(0, 10, n)
x2 = np.random.uniform(0, 5, n)
x3 = np.random.uniform(1, 3, n)

# True function (only some terms actually matter)
y = (
    3
    + 2*x1
    + 1.5*x2
    - 1*x3
    + 0.5*x1**2
    + 0.3*x1*x2
    + np.random.randn(n) * 2
)

# -------------------------
# 2️⃣ Build Polynomial Features
# -------------------------
X = np.column_stack((
    np.ones(n),
    x1,
    x2,
    x3,
    x1**2,
    x2**2,
    x3**2,
    x1*x2,
    x1*x3,
    x2*x3
))

# -------------------------
# 3️⃣ Scale Features (except bias)
# -------------------------
X_features = X[:, 1:]
mean = X_features.mean(axis=0)
std = X_features.std(axis=0)

X_scaled = np.column_stack((
    np.ones(n),
    (X_features - mean) / std
))

y = y.reshape(-1, 1)

# -------------------------
# 4️⃣ Initialize
# -------------------------
theta = np.zeros((X_scaled.shape[1], 1))

alpha = 0.01
lambda_ = 0.1
epochs = 5000
m = n

# -------------------------
# 5️⃣ Lasso Gradient Descent
# -------------------------
for _ in range(epochs):

    # Prediction
    y_pred = X_scaled @ theta

    # Error
    error = y_pred - y

    # Normal gradient (MSE part)
    gradient = (2/m) * (X_scaled.T @ error)

    # Lasso shrinkage (do NOT touch bias)
    gradient[1:] += lambda_ * np.sign(theta[1:])

    # Update
    theta = theta - alpha * gradient

# -------------------------
# 6️⃣ See Final Weights
# -------------------------
print("Learned Weights:")
print(theta)
