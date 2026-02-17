import numpy as np

# -------------------------------
# STEP 0: Reproducibility
# -------------------------------
np.random.seed(0)

# -------------------------------
# STEP 1: Generate synthetic data
# -------------------------------
n = 100

# Three input features
x1 = np.random.uniform(0, 10, n)
x2 = np.random.uniform(0, 5, n)
x3 = np.random.uniform(1, 3, n)

# True underlying function (unknown to model)
y = (
    3
    + 2*x1
    + 1.5*x2
    - 1*x3
    + 0.5*x1**2
    + 0.8*x2**2
    + 0.3*x1*x2
    + np.random.randn(n) * 2   # noise
)

# -------------------------------
# STEP 2: Build polynomial features (degree = 2)
# Total features = 10
# -------------------------------
X = np.column_stack((
    np.ones(n),      # bias
    x1,              # linear
    x2,
    x3,
    x1**2,           # squared
    x2**2,
    x3**2,
    x1*x2,           # interaction
    x1*x3,
    x2*x3
))

# -------------------------------
# STEP 3: Train / Validation split
# -------------------------------
indices = np.arange(n)
np.random.shuffle(indices)

train_size = int(0.7 * n)
train_idx = indices[:train_size]
val_idx = indices[train_size:]

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

# -------------------------------
# STEP 4: Feature scaling (IMPORTANT)
# Do NOT scale bias column
# -------------------------------
X_train_features = X_train[:, 1:]
mean = X_train_features.mean(axis=0)
std = X_train_features.std(axis=0)

X_train_scaled = np.column_stack((
    np.ones(len(X_train)),
    (X_train_features - mean) / std
))

X_val_scaled = np.column_stack((
    np.ones(len(X_val)),
    (X_val[:, 1:] - mean) / std
))

# Convert y to column vectors
y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

# -------------------------------
# STEP 5: Initialize parameters
# -------------------------------
theta = np.zeros((X_train_scaled.shape[1], 1))

alpha = 0.01       # learning rate
lambda_ = 0.1      # ridge penalty
epochs = 5000
m = len(y_train)

# -------------------------------
# STEP 6: Ridge Gradient Descent
# -------------------------------
for _ in range(epochs):

    # Prediction
    y_pred = X_train_scaled @ theta

    # Error
    error = y_pred - y_train

    # Gradient of MSE
    gradient = (2/m) * (X_train_scaled.T @ error)

    # Ridge penalty (DO NOT penalize bias)
    gradient[1:] += 2 * lambda_ * theta[1:]

    # Update weights
    theta = theta - alpha * gradient

# -------------------------------
# STEP 7: Evaluation
# -------------------------------
def mse(X, y, theta):
    preds = X @ theta
    return np.mean((preds - y) ** 2)

train_error = mse(X_train_scaled, y_train, theta)
val_error = mse(X_val_scaled, y_val, theta)

print("Training MSE:", train_error)
print("Validation MSE:", val_error)

# -------------------------------
# STEP 8: Predict on unseen sample
# -------------------------------
x1_new, x2_new, x3_new = 6, 2, 1.5

X_new = np.array([
    1,
    x1_new,
    x2_new,
    x3_new,
    x1_new**2,
    x2_new**2,
    x3_new**2,
    x1_new*x2_new,
    x1_new*x3_new,
    x2_new*x3_new
]).reshape(1, -1)

# Scale using TRAIN mean/std
X_new_scaled = np.column_stack((
    np.ones(1),
    (X_new[:, 1:] - mean) / std
))

y_pred_new = X_new_scaled @ theta
print("Prediction for new input:", y_pred_new[0, 0])
