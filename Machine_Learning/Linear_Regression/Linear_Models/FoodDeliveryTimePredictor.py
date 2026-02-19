import numpy as np

#Distance in km
distance = np.array([2.5,6.0,1.2,8.5,3.8,5.2,1.8,7.0,4.5,9.2,2.0,6.5,3.2,7.8,4.0,5.8,1.5,8.0,3.5,6.8,2.2,5.5,4.2,9.0,2.8,7.2,3.0,6.2,4.8,8.2])

#Prep time in min
prep_time= np.array([10,20,8,25,12,18,7,22,15,28,9,19,11,24,14,17,6,26,13,21,10,16,14,27,11,23,12,18,15,25])

#target time of delivery
delivery_time = np.array([18,38,12,52,24,34,14,45,29,58,15,40,21,50,27,35,11,54,23,43,17,32,26,56,19,47,20,37,30,53])
print(distance.shape)
print(prep_time.shape)
print(delivery_time.shape)

ones = np.ones(len(distance))
X = np.column_stack((ones,distance,prep_time))

y = delivery_time.reshape(-1,1)

print(X.shape)
print(y.shape)
print(X)
print(y)

X_features = X[:,1:]
mean = X_features.mean(axis= 0)

print("Mean :",mean)

std = X_features.std(axis= 0)
print("Std:", std)

X_scaled_features = (X_features - mean) / std

print ("X scaled Features:", X_scaled_features)

X_Scaled = np.column_stack((np.ones(len(X_scaled_features)),X_scaled_features))
print (X_Scaled.shape)
print(X_Scaled[:5])

theta = np.zeros((3,1))
print(theta)

alpha = 0.05
epochs = 3000

n = len(y)

for i in range(epochs):
    y_pred = X_Scaled @ theta
    error = y_pred - y
    gradient = (2 / n) * (X_Scaled.T @ error)
    theta = theta - alpha * gradient
print (theta)





bias = theta[0][0]
w_distance = theta[1][0]
w_prep = theta[2][0]

new_input = np.array([7.0,15.0])
new_input_Scaled = (new_input - mean) / std

X_new = np.array([1,new_input_Scaled[0],new_input_Scaled[1]])

prediction = X_new @ theta 
print ("Prediction final: ", prediction)





