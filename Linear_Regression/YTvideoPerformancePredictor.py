import numpy as np
import matplotlib.pyplot as plt 
x = np.array([3.2,5.8,2.1,7.4,4.5,6.2,1.8,5.1,3.9,8.5,4.2,2.8,6.8,3.5,7.1,2.4,5.5,4.8,6.5,3.1,7.8,2.6,5.3,4.1,6.9,3.7,8.1,2.2,5.9,4.6])
y = np.array([12000,28000,8500,42000,19000,33000,7000,24000,16000,51000,18000,11000,38000,14500,44000,9500,26000,21000,35000,13000,47000,10000,25000,17500,40000,15000,49000,8000,31000,20000])
x_mean = np.mean(x)
y_mean = np.mean(y)
w = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)

b = y_mean - w * x_mean
y_pred = w * x + b
loss = np.mean((y - y_pred) ** 2)
print ("Loss:",loss)

arjun_input = 8
predicted_views = w * arjun_input + b

print("Views for 8% CTR of arjun video predicted :", predicted_views)
x_line = np.linspace(min(x),max(x), 100)
y_line = w * x_line + b
plt.scatter(x,y)
plt.plot(x_line,y_line)
plt.xlabel("CTR (%)")
plt.ylabel("CTR vs Views (Actua Data)")
plt.title("LInear Regression: CTR vs Views")
plt.show()



print("Final w:", w)
print("Final b:", b)
