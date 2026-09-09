import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("Datasets/EMG.csv", delimiter=' ')
classes = np.unique(data[:,-1])
classes = [1,2,3, 4,5]
nomes = [
        'neutro',
         'sorriso',
         'sobrancelhas levantadas',
         'surpreso',
         'rabugento']
C = len(classes)
X_treino = np.empty((0, 2))
Y_treino = np.empty((0, C))
for i,classe in enumerate(classes):
    X_classe = data[data[:,-1]== classe ,:-1]
    X_treino = np.vstack((
        X_treino, X_classe
    ))
    y = -np.ones((1, C))
    y[0, i] = 1
    Y_treino = np.vstack((
        Y_treino, np.tile(y, (X_classe.shape[0],1))
    ))
    
    plt.scatter(X_classe[:,0], X_classe[:,1], edgecolors='k',
                label=nomes[i])

#Pesos
X_treino = np.hstack((
    np.ones((X_treino.shape[0],1)), X_treino
))

W =  np.linalg.pinv(X_treino)@Y_treino  


x1 = np.linspace(-300,5000,1000)
x2 = -W[1,0]/W[2,0]*x1 - W[0,0]/W[2,0]
plt.plot(x1,x2)
x2 = -W[1,1]/W[2,1]*x1 - W[0,1]/W[2,1]
plt.plot(x1,x2)
x2 = -W[1,2]/W[2,2]*x1 - W[0,2]/W[2,2]


X1,X2 = np.meshgrid(x1,x1)
X_plot = np.concatenate((
    np.ones((X1.shape[0],X1.shape[1],1)),
    X1[:,:,None], 
    X2[:,:,None]
),axis=2)

Y_hat = X_plot @ W



x_novo = np.array([[1, 1153, 902]])
y_hat = x_novo @ W


plt.scatter(1153, 902,marker='x')
j = np.argmax(Y_hat, axis=2)
plt.plot(x1,x2)
plt.contourf(X1,X2, j, alpha=.3)
plt.xlim(-200,4095)
plt.ylim(-200,4095)
plt.legend()
plt.show()


bp = 1


def classificar_mqo():
    return None

def classificar_gaussiano_tradicional():
    return None

def classificar_gaussiano_cov_total():
    return None


def classificar_gaussiano_cov_agregada():
    return None


def classificar_bayes_ingenuo():
    return None


def classificar_gaussiano_regularizado():
    return None