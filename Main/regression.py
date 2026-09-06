import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("../Datasets/china_gdp.csv", delimiter=',', skiprows=1)

#1

plt.figure(0)
plt.scatter(data[:, 0], data[:, 1])
plt.xlabel("Ano")
plt.ylabel("PIB")
# plt.show()

#2

x = data[:, 0].reshape(data.shape[0],1)
y = data[:, 1].reshape(data.shape[0],1)

#3 MQO Tradicional

X = np.concatenate((np.ones((len(x), 1)), x),axis=1)
B = np.linalg.pinv(X.T@X)@X.T@y

x_axis = np.linspace(data[0,0], data[-1,0], 100)
x_axis = x_axis.reshape(len(x_axis), 1)
X_new = np.concatenate((np.ones((len(x_axis), 1)), x_axis),axis=1)
Y_pred = X_new@B

plt.plot(x_axis, Y_pred)
# plt.show()

#3.1 MQO Regularizado Normalizado

plt.figure(1)
plt.scatter(data[:, 0], data[:, 1])
plt.xlabel("Ano")
plt.ylabel("PIB")

x_normalized = (x - x.mean()) / x.std()
X = np.concatenate((np.ones((len(x_normalized), 1)), x_normalized), axis=1)

I = np.eye(X.shape[1])
I[0,0] = 0
lamb = 100
B = np.linalg.pinv((X.T@X) + (lamb*I))@X.T@y

x_axis = np.linspace(data[0,0], data[-1,0], 100)
x_axis = x_axis.reshape(len(x_axis), 1)
x_axis_normalized = (x_axis - x.mean()) / x.std()
X_new = np.concatenate((np.ones((len(x_axis_normalized), 1)), x_axis_normalized),axis=1)
Y_pred = X_new@B

plt.plot(x_axis, Y_pred)

#3.2 MQO Regularizado Não Normalizado

plt.figure(2)
plt.scatter(data[:, 0], data[:, 1])
plt.xlabel("Ano")
plt.ylabel("PIB")

I = np.eye(X.shape[1])
I[0,0] = 0
lamb = 1000000
B = np.linalg.pinv((X.T@X) + (lamb*I))@X.T@y

x_axis = np.linspace(data[0,0], data[-1,0], 100)
x_axis = x_axis.reshape(len(x_axis), 1)
X_new = np.concatenate((np.ones((len(x_axis), 1)), x_axis),axis=1)
Y_pred = X_new@B

plt.plot(x_axis, Y_pred)
plt.show()

print()