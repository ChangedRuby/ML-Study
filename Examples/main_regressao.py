import numpy as np
import matplotlib.pyplot as plt
from cirillo_kitlearn import LinearRegression


data = np.loadtxt("../Datasets/aerogerador.dat", delimiter='\t')
X = data[:,0].reshape(data.shape[0],1)
Y = data[:,-1:]
#Padronização (Z-score)
media_x = np.mean(X)
desv_pad_x = np.std(X)
X = (X-media_x)/desv_pad_x
Y = (Y-np.mean(Y))/np.std(Y)

lr = LinearRegression(X,Y,fit_intercept=True)

lr.fit()

Y_pred = lr.predict(X)
E = Y - Y_pred

fig = plt.figure(1)
plt.scatter(X[:], Y[:], c='r', edgecolors='k',)
plt.xlabel("Velocidade do Vento")
plt.ylabel("Potência Gerada")
plt.grid(True)
plt.xlim(-5.3, 3.3)
plt.ylim(-3, 2.3)
x = np.linspace(-6,5, 200)
y_pred = lr.predict(x.reshape(len(x),1))




plt.plot(x, y_pred, c='yellow',lw=3)



plt.figure(2)
plt.hist(E,bins=30, color='teal',edgecolor='k',density=True)
mu = np.mean(E)
std = np.std(E)
x = np.linspace(-2,2,2000)
y = 1/(np.sqrt(2*std)) * np.exp(-(((x-mu)**2)/(2*std)))
plt.plot(x,y,c='orange')
####### para fins didáticos ####

b0 = np.linspace(-200,200,300)
b1 = np.linspace(-150,150,300)
B0,B1 = np.meshgrid(b0,b1)
Y_pred_plot = B0[:,:,None] + B1[:,:,None]*X[:,-1]
E = Y[:,-1] - Y_pred_plot
J = np.sum(E**2, axis=2)
fig = plt.figure(3)
ax = fig.add_subplot(projection='3d')
ax.plot_surface(B0,B1, J,alpha=.7,cmap='turbo')
# ax.scatter(lr.beta[0],lr.beta[1],0)
plt.show()

#SSE, MSE, R^2



#Acurácia, Sensibilidade, Especificidade e F1-Score

bp = 1