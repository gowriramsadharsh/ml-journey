##
import numpy as np

np.random.seed(0)

n = 100  # number of samples

x1 = np.random.uniform(0, 10, n) #Distance random 
x2 = np.random.uniform(0, 5, n)  #Prep time
x3 = np.random.uniform(1, 3, n)  #Traffic 


y = (
    3
    + 2*x1
    + 1.5*x2
    - 1*x3
    + 0.5*x1**2
    + 0.8*x2**2
    + 0.3*x1*x2
    + np.random.randn(n) * 2
)

print(y.shape)
X = np.column_stack((
    np.ones(n),      # Bias
    x1,              # Linear terms
    x2,
    x3,
    x1**2,           # Squared terms
    x2**2,
    x3**2,
    x1*x2,           # Interaction terms
    x1*x3,
    x2*x3
))

# Remove bias column before scaling
X_features = X[:, 1:]

# Compute mean and standard deviation
mean = X_features.mean(axis=0)
std = X_features.std(axis=0)

# Scale features
X_scaled_features = (X_features - mean) / std

# Add bias back
X_scaled = np.column_stack((np.ones(n), X_scaled_features))

#Initialize zeroes
theta = np.zeros((10, 1))

alpha = 0.01    #learning rate
epochs = 5000   #iterations
n = X_scaled.shape[0]  #number of data samples


for _ in range(epochs):

    y_pred = X_scaled @ theta
    
    error = y_pred - y.reshape(-1, 1)
    
    gradient = (2/n) * (X_scaled.T @ error)
    
    theta = theta - alpha * gradient


    # New unseen sample
x1_new = 6
x2_new = 2
x3_new = 1.5


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
])


X_new = X_new.reshape(1, -1)


X_new_features = X_new[:, 1:]
X_new_scaled_features = (X_new_features - mean) / std

X_new_scaled = np.column_stack((np.ones(1), X_new_scaled_features))


y_pred_new = X_new_scaled @ theta
print(y_pred_new)


y_true = (
    3
    + 2*x1_new
    + 1.5*x2_new
    - 1*x3_new
    + 0.5*x1_new**2
    + 0.8*x2_new**2
    + 0.3*x1_new*x2_new
)


print("Predicted:", y_pred_new)
print("True:", y_true)
