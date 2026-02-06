import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate input
X = np.linspace(0, 10, 50)

# True relationship (unknown to model)
y = 0.5 * X**2 + X + 3 + np.random.randn(50) * 4

plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Raw Data (Unknown Relationship)")
plt.show()

print(len(X))
# Shuffle indices
indices = np.arange(len(X))
np.random.shuffle(indices)

train_size = int(0.7 * len(X))
train_idx = indices[:train_size]
val_idx = indices[train_size:]

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

