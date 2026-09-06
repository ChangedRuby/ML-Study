# N = 951 (Amostras, observações)
# p = 2


import numpy as np
import matplotlib.pyplot as plt
from cirillo_kitlearn import PolynomialRegression

data = np.loadtxt("Solubilidade.csv",delimiter=',')
N, p = data.shape
X = data[:,:-1]
Y = data[:,-1:]
rodadas = 200

for i in range(rodadas):
    
    idx = np.random.permutation(N)
    Xr = np.copy(X)[idx,:]
    Yr = np.copy(Y)[idx,:]
    
    X_treino = Xr[:int(.8*N),:]
    Y_treino = Yr[:int(.8*N),:]
    
    X_teste = Xr[int(.8*N):,:]
    Y_teste = Yr[int(.8*N):,:]
    
    pr = PolynomialRegression(X_treino, Y_treino, q = 1)
    pr.fit()
    
    Y_pred = pr.predict(X_treino    )
    E = Y_teste - Y_pred
    SSE = np.sum(E**2)
    print(f"SSE: {SSE}")
    MSE = np.mean(E**2)
    print(f"MSE: {MSE}")
    y_bar = np.mean(Y_teste)
    SST = np.sum((Y_teste - y_bar)**2)
    R2 = 1 - SSE/SST
    bp=1
    
    
    
    
    
    
    
    
    
    
    
    
    if i == 0:
        x1 = np.linspace(0,35,300)
        x2 = np.linspace(10,700,300)
        b0 = np.mean(Y_treino)
        X1,X2 = np.meshgrid(x1,x2)
        X_plot = np.concatenate((
            X1[:,:, None],
            X2[:,:, None],
        ),axis=2)
        Y_pred = pr.predict(X_plot)
        B0 = np.ones(X1.shape)*b0
        fig = plt.figure(1)
        ax = fig.add_subplot(1,2,1,projection='3d')
        ax.scatter(X_treino[:,0],
                   X_treino[:,1],
                   Y_treino[:,0], c='red',
                   edgecolor='k')
        ax.set_title("Dados de treinamento")
        ax.plot_surface(X1,X2,B0)
        ax.set_zlim(-6,8)
        ax = fig.add_subplot(1,2,2,projection='3d')
        ax.scatter(X_teste[:,0],
                   X_teste[:,1],
                   Y_teste[:,0], c='green',
                   edgecolor='k')
        ax.set_title("Dados de teste")
        
    
        plt.show()
    bp=1



bp = 1