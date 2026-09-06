import numpy as np
import matplotlib.pyplot as plt

x = np.array([480 ,500 ,380 ,1100,1100,230 ,490 ,250 ,300 ,510])
x = x.reshape((len(x),1))
y = np.array([180,150,170,350,460,60,240,90,110,250])
y = y.reshape((len(y),1))

X = np.concatenate((np.ones((len(x),1)), x),axis=1)
B = np.linalg.pinv(X.T@X)@X.T@y

x_axis = np.linspace(0,1200,100)
x_axis = x_axis.reshape(len(x_axis),1)
X_new = np.concatenate((np.ones((len(x_axis),1)), x_axis),axis=1)
Y_pred = X_new@B

plt.plot(x_axis, Y_pred)
plt.scatter(x,y, color="orange")
plt.show()
print()