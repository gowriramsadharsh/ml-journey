import numpy as np
Ram_gb = np.array([4,8,4,16,8,8,4,16,8,16,4,8,4,16,8,8,4,16,8,16,4,8,4,16,8,8,4,16,8,16])
Storage_gb = np.array([256,512,128,512,256,512,256,1024,256,512,128,512,256,1024,256,512,128,512,256,1024,256,512,128,512,256,512,256,1024,256,512])
Processor_ghz = np.array([2.1,2.8,1.8,3.2,2.4,3.0,2.0,3.5,2.6,3.0,1.6,2.8,2.2,3.4,2.5,2.9,1.9,3.1,2.3,3.6,2.0,2.7,1.7,3.3,2.4,3.0,2.1,3.5,2.6,3.2])
Price_inr= np.array([28000,45000,22000,72000,38000,52000,26000,95000,42000,68000,20000,48000,29000,88000,40000,50000,23000,70000,36000,98000,25000,46000,21000,75000,39000,53000,27000,92000,43000,73000])

#check each feature size are equal or not
#print(Ram_gb.shape)
#print(Storage_gb.shape)
#print(Processor_ghz.shape)
#print(Price_inr.shape)

one = np.ones(len(Ram_gb))


X = np.column_stack((one,Ram_gb,Storage_gb,Processor_ghz))
Y = Price_inr.reshape(-1,1)
#print(X.shape)
#print(Y.shape)
#print(X)
#print(Y)
X_features = X[:,1:]
mean = X_features.mean(axis=0)

#print("Mean =", mean)

standard_dev = X_features.std(axis=0)
#print("Std = " , standard_dev)


X_Scaled_Features = (X_features - mean)/standard_dev

#print("Scaled Features = ", X_Scaled_Features)

X_Scaled = np.column_stack((np.ones(len(X_Scaled_Features)),X_Scaled_Features))
#print(X_Scaled.shape)
#print(X_Scaled)
theta = np.zeros((4,1))
print(theta)

alpha = 0.05
ephoes = 3000
n = len(Y)

for i in range(ephoes):
    y_pred = X_Scaled @ theta
    error = y_pred - Y
    gradient = (2/n) * (X_Scaled.T @ error)
    theta = theta - alpha * gradient

print(theta)

bias = theta[0][0]
w_ram = theta[1][0]
w_storage= theta[2][0]
w_processor = theta[3][0]

new_input = np.array([16,512,3.2])
new_input_Scaled = (new_input - mean) / standard_dev

X_new = np.array([1,new_input_Scaled[0],new_input_Scaled[1],new_input_Scaled[2]])

prediction = X_new @ theta 
print ("Prediction final: ", prediction)

