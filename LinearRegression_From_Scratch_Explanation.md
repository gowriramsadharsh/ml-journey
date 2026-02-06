# Multivariate Linear Regression From Scratch — Line‑by‑Line Explanation

This document explains **every single line** of your code **exactly as written**, with **intuition**, **math meaning**, **shape awareness**, and **why it is needed**. Nothing is skipped.

---

## 0. Goal of the Program (Big Picture)

The program learns a mathematical rule that maps **laptop specifications → price** using:

* manual feature scaling
* gradient descent
* matrix multiplication

The learned model is:

[\text{Price} = \theta_0 + \theta_1(\text{RAM}) + \theta_2(\text{Storage}) + \theta_3(\text{Processor})]

---

## 1. Import NumPy

```python
import numpy as np
```

**Why this exists**

* Linear regression is matrix math
* NumPy provides fast vectors, matrices, and linear algebra
* Without NumPy, gradient descent would be impractical

---

## 2. Input Feature Arrays

```python
Ram_gb = np.array([...])
Storage_gb = np.array([...])
Processor_ghz = np.array([...])
Price_inr = np.array([...])
```

**Intuition**

* Each array represents one real‑world feature
* Index `i` across all arrays describes **one laptop**

Example:

> Laptop `i` has `Ram_gb[i]`, `Storage_gb[i]`, `Processor_ghz[i]` → costs `Price_inr[i]`

---

## 3. Bias Column (Intercept)

```python
one = np.ones(len(Ram_gb))
```

**Why this is needed**

* Allows the model to learn a **base price**
* Prevents forcing predictions through zero

Mathematically this represents **θ₀**.

---

## 4. Construct the Design Matrix X

```python
X = np.column_stack((one, Ram_gb, Storage_gb, Processor_ghz))
```

**Resulting structure**

Each row becomes:

```
[1, RAM, Storage, Processor]
```

**Meaning**

* One row = one equation
* One column = one parameter

This is how math sees the dataset.

---

## 5. Target Vector y

```python
Y = Price_inr.reshape(-1,1)
```

**Why reshape?**

* Ensures shape `(n, 1)`
* Required for correct matrix multiplication

Avoids silent broadcasting bugs.

---

## 6. Separate Features from Bias

```python
X_features = X[:,1:]
```

**Why?**

* Bias column must **never be scaled**
* Only real features are normalized

---

## 7. Compute Feature Means

```python
mean = X_features.mean(axis=0)
```

**Meaning**

* Computes mean RAM, mean Storage, mean Processor
* Used to center features around zero

---

## 8. Compute Feature Standard Deviation

```python
standard_dev = X_features.std(axis=0)
```

**Meaning**

* Measures how spread each feature is
* Prevents large‑scale features from dominating learning

---

## 9. Feature Scaling (Standardization)

```python
X_Scaled_Features = (X_features - mean) / standard_dev
```

**What this does**

* Mean → 0
* Standard deviation → 1

**Why this is critical**

* Balances gradients
* Enables stable gradient descent
* Allows reasonable learning rate

---

## 10. Re‑add Bias Column

```python
X_Scaled = np.column_stack((np.ones(len(X_Scaled_Features)), X_Scaled_Features))
```

**Why after scaling?**

* Bias must remain exactly `1`
* Scaling bias would break the model

---

## 11. Initialize Parameters θ

```python
theta = np.zeros((4,1))
```

**Meaning**

* Start with no assumptions
* Model will learn weights automatically

Structure:

```
θ₀ → bias
θ₁ → RAM weight
θ₂ → Storage weight
θ₃ → Processor weight
```

---

## 12. Hyperparameters

```python
alpha = 0.05
epochs = 3000
n = len(Y)
```

**Meaning**

* `alpha`: step size
* `epochs`: number of learning iterations
* `n`: number of samples (for averaging gradients)

---

## 13. Gradient Descent Loop

```python
for i in range(epochs):
```

Each loop = one learning step.

---

### 13.1 Prediction

```python
y_pred = X_Scaled @ theta
```

Computes:
[\hat{y} = X\theta]

Meaning: model guesses prices for all laptops.

---

### 13.2 Error Calculation

```python
error = y_pred - Y
```

Meaning:

* Positive → overprediction
* Negative → underprediction

---

### 13.3 Gradient Computation

```python
gradient = (2/n) * (X_Scaled.T @ error)
```

**What this computes**

* How responsible each feature is for the error
* One gradient per parameter

Derived from MSE cost function.

---

### 13.4 Parameter Update

```python
theta = theta - alpha * gradient
```

**Meaning**

* Move parameters in direction that reduces error
* Repeated → convergence

---

## 14. Extract Learned Parameters

```python
bias = theta[0][0]
w_ram = theta[1][0]
w_storage = theta[2][0]
w_processor = theta[3][0]
```

**Why extract?**

* Interpret model
* Understand feature influence
* Debug learning

Each value shows effect per **1 standard deviation** increase.

---

## 15. Prepare New Input

```python
new_input = np.array([16,512,3.2])
```

Represents a new laptop.

---

## 16. Scale New Input (CRITICAL)

```python
new_input_Scaled = (new_input - mean) / standard_dev
```

**Why required**

* Model expects scaled inputs
* Must use training mean/std

Production rule.

---

## 17. Build Input Row for Prediction

```python
X_new = np.array([1, new_input_Scaled[0], new_input_Scaled[1], new_input_Scaled[2]])
```

Matches training format:

```
[1, RAM_scaled, Storage_scaled, Processor_scaled]
```

---

## 18. Final Prediction

```python
prediction = X_new @ theta
```

Expands to:

[
\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3
]

This is the **learned price**.

---

## 19. Key Takeaway

You implemented:

* real feature scaling
* real gradient descent
* real inference logic

This is **foundational machine learning**, not library usage.

---

## 20. Final Mental Model

> Training: learn θ
> Prediction: apply θ
> Scaling: keeps learning fair

---

End of document.
