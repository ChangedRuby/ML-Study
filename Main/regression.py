import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("../Datasets/china_gdp.csv", delimiter=',', skiprows=1)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!1

plt.figure(0)
plt.scatter(data[:, 0], data[:, 1])
plt.xlabel("Ano")
plt.ylabel("PIB")
# plt.show()

#!!!!!!!!!!!!!!!!!!!!!!!!!!!2

x = data[:, 0].reshape(data.shape[0],1)
y = data[:, 1].reshape(data.shape[0],1)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!3 MQO Tradicional

X = np.concatenate((np.ones((len(x), 1)), x),axis=1)
B = np.linalg.pinv(X.T@X)@X.T@y

x_axis = np.linspace(data[0,0], data[-1,0], 100)
x_axis = x_axis.reshape(len(x_axis), 1)
X_testing = np.concatenate((np.ones((len(x_axis), 1)), x_axis), axis=1)
Y_pred = X_testing @ B

plt.plot(x_axis, Y_pred)
# plt.show()

#!!!!!!!!!!!!!!!!!!!!!!!!!!!3.1 MQO Regularizado Normalizado

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
X_testing = np.concatenate((np.ones((len(x_axis_normalized), 1)), x_axis_normalized), axis=1)
Y_pred = X_testing @ B

plt.plot(x_axis, Y_pred)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!3.2 MQO Regularizado Não Normalizado

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
X_testing = np.concatenate((np.ones((len(x_axis), 1)), x_axis), axis=1)
Y_pred = X_testing @ B

plt.plot(x_axis, Y_pred)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!4 MQO Polinomial

plt.figure(3)
plt.scatter(data[:, 0], data[:, 1])
plt.xlabel("Ano")
plt.ylabel("PIB")

x_normalized = (x - x.mean()) / x.std()
X = np.concatenate((np.ones((len(x_normalized), 1)), x_normalized), axis=1)
q = 6

for i, n in enumerate(range(2, q + 1)):
    X = np.concatenate((X, x_normalized**n), axis=1)

I = np.eye(X.shape[1])
I[0,0] = 0
lamb = 1
B = np.linalg.pinv((X.T@X) + (lamb*I))@X.T@y

x_axis = np.linspace(data[0,0], data[-1,0], 100)
x_axis = x_axis.reshape(len(x_axis), 1)
x_axis_normalized = (x_axis - x.mean()) / x.std()
X_testing = np.concatenate((np.ones((len(x_axis_normalized), 1)), x_axis_normalized), axis=1)

for i, n in enumerate(range(2, q + 1)):
    X_testing = np.concatenate((X_testing, x_axis_normalized ** n), axis=1)

Y_pred = X_testing @ B

plt.plot(x_axis, Y_pred)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!4 Estrategia de poda

plt.figure(4)
plt.xlabel("Ano")
plt.ylabel("PIB")

indicies = np.random.permutation(len(x))
x_shuffled = x[indicies]
y_shuffled = y[indicies]
x_training = x_shuffled[:int(.8*len(x_shuffled))]
x_testing = x_shuffled[int(.8*len(x_shuffled)):]
x_training_normalized = (x_training - x_training.mean()) / x_training.std()
x_testing_normalized = (x_testing - x_training.mean()) / x_training.std()
y_training = y_shuffled[:int(.8*len(y_shuffled))]
y_testing = y_shuffled[int(.8*len(y_shuffled)):]

plt.scatter(x_training, y_training, color="blue")

X = np.concatenate((np.ones((len(x_training_normalized), 1)), x_training_normalized), axis=1)
q = 6

for i, n in enumerate(range(2, q + 1)):
    X = np.concatenate((X, x_training_normalized**n), axis=1)

I = np.eye(X.shape[1])
I[0,0] = 0
lamb = 1
B = np.linalg.pinv((X.T@X) + (lamb*I))@X.T@y_training

####### TESTA COM OS DADOS DE TESTE ########
X_testing = np.concatenate((np.ones((len(x_testing_normalized), 1)), x_testing_normalized), axis=1)

for i, n in enumerate(range(2, q + 1)):
    X_testing = np.concatenate((X_testing, x_testing_normalized**n), axis=1)

Y_pred = X_testing@B

R_squared = 1 - (np.sum((y_testing - Y_pred)**2) / np.sum((y_testing - np.mean(y_testing))**2))
###################

####### CRIA O PLOT COM 100 DADOS COM LINSPACE ########
# função antiga quando o plot era os próprios dados de teste
# ordered_indicies = np.argsort(x_testing, axis=0).flatten()
# criar o plot dos dados de teste
# plt.plot(x_testing[ordered_indicies], Y_pred[ordered_indicies])

x_plot = np.linspace(np.union1d(x_training, x_testing).min(), np.union1d(x_training, x_testing).max(), 100)
x_plot = x_plot.reshape(len(x_plot), 1)
x_plot_normalized = (x_plot - x_training.mean()) / x_training.std()
X_plot = np.concatenate((np.ones((len(x_plot_normalized), 1)), x_plot_normalized), axis=1)

for i, n in enumerate(range(2, q + 1)):
    X_plot = np.concatenate((X_plot, x_plot_normalized**n), axis=1)

Y_pred_plot = X_plot@B
###################

plt.title(f"R2 = {R_squared:.8f}")
plt.scatter(x_testing, y_testing, color="red")
plt.scatter(x_testing, Y_pred, marker="x", color="orange")
plt.plot(x_plot, Y_pred_plot)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!4.1 Estrategia de poda com média do R2

plt.figure(5)
plt.xlabel("p")
plt.ylabel("R²")

final_R_squared = []
start_q = 20
end_q = 2
for q in range(start_q, end_q - 1, -1):
    tested_R_squared = []
    testing_amount = 100
    for i in range(testing_amount):
        indicies = np.random.permutation(len(x))
        x_shuffled = x[indicies]
        y_shuffled = y[indicies]
        x_training = x_shuffled[:int(.8*len(x_shuffled))]
        x_testing = x_shuffled[int(.8*len(x_shuffled)):]
        x_training_normalized = (x_training - x_training.mean()) / x_training.std()
        x_testing_normalized = (x_testing - x_training.mean()) / x_training.std()
        y_training = y_shuffled[:int(.8*len(y_shuffled))]
        y_testing = y_shuffled[int(.8*len(y_shuffled)):]

        X = np.concatenate((np.ones((len(x_training_normalized), 1)), x_training_normalized), axis=1)

        for i, n in enumerate(range(2, q + 1)):
            X = np.concatenate((X, x_training_normalized**n), axis=1)

        I = np.eye(X.shape[1])
        I[0,0] = 0
        lamb = 1
        B = np.linalg.pinv((X.T@X) + (lamb*I))@X.T@y_training

        ####### TESTA COM OS DADOS DE TESTE ########
        X_testing = np.concatenate((np.ones((len(x_testing_normalized), 1)), x_testing_normalized), axis=1)

        for i, n in enumerate(range(2, q + 1)):
            X_testing = np.concatenate((X_testing, x_testing_normalized**n), axis=1)

        Y_pred = X_testing@B

        R_squared = 1 - (np.sum((y_testing - Y_pred)**2) / np.sum((y_testing - np.mean(y_testing))**2))
        tested_R_squared.append(R_squared)
        ###################

    final_R_squared.append(np.mean(tested_R_squared))

# print(f"R_squared para cada p: {final_R_squared}")
for i, r in enumerate(final_R_squared):
    print(f"para o p = {start_q - i}: R_squared = {r}")

####### CRIA O HIST COM O R_squared DE CADA p ########
plt.bar(range(start_q, end_q - 1, -1), final_R_squared)
###################

plt.show()

print()