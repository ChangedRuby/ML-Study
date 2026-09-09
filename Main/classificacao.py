import numpy as np
import matplotlib.pyplot as plt
import time

data = np.loadtxt("Datasets/EMG.csv", delimiter=' ')

sensor1 = data[:, 0]
sensor2 = data[:, 1]
classes = data[:, 2]

X = data[:, 0:2]

Y = np.zeros((len(classes), 5))

for i in range(len(classes)):
    Y[i, int(classes[i]) - 1] = 1

for classe in range(1, 6):

    indices = classes == classe

    plt.scatter(
        sensor1[indices],
        sensor2[indices],
        label=f"Classe {classe}",
        alpha=0.5
    )

plt.xlabel("Sensor 1 - Corrugador")
plt.ylabel("Sensor 2 - Zigomático")
plt.title("Distribuição das classes")
plt.legend()
plt.grid()


X_mqo = np.column_stack((np.ones(X.shape[0]), X))
W = np.linalg.pinv(X_mqo) @ Y

Y_pred = X_mqo @ W
pred_classes = np.argmax(Y_pred, axis=1) + 1

accuracy = np.mean(pred_classes == classes)

print("Dimensão X:", X.shape)
print("Dimensão Y:", Y.shape)
print("Dimensão W:", W.shape)
print("Acurácia MQO:", accuracy)

# Ver MQO Tradicional

x1_min, x1_max = sensor1.min(), sensor1.max()
x2_min, x2_max = sensor2.min(), sensor2.max()

xx1, xx2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 300),
    np.linspace(x2_min, x2_max, 300)
)

X_grade = np.column_stack((
    np.ones(xx1.ravel().shape[0]),
    xx1.ravel(),
    xx2.ravel()
))

Y_grade = X_grade @ W
classes_grade = np.argmax(Y_grade, axis=1) + 1
classes_grade = classes_grade.reshape(xx1.shape)

plt.figure(figsize=(8, 6))
plt.contourf(
    xx1,
    xx2,
    classes_grade,
    alpha=0.25
)
for classe in range(1, 6):
    indices = classes == classe

    plt.scatter(
        sensor1[indices],
        sensor2[indices],
        label=f"Classe {classe}",
        alpha=0.3
    )

plt.xlabel("Sensor 1 - Corrugador")
plt.ylabel("Sensor 2 - Zigomático")
plt.title(f"MQO Tradicional - Acurácia: {accuracy:.4f}")
plt.legend()
plt.grid()


# MQO Regularizado

lambda_reg = 0.9 #mudar?

I = np.eye(X_mqo.shape[1])

W_reg = np.linalg.pinv(
    X_mqo.T @ X_mqo + lambda_reg * I
) @ X_mqo.T @ Y


Y_pred_reg = X_mqo @ W_reg

pred_classes_reg = np.argmax(Y_pred_reg, axis=1) + 1

accuracy_reg = np.mean(pred_classes_reg == classes)
print("Acurácia MQO regularizado:", accuracy_reg)

x1_min, x1_max = sensor1.min(), sensor1.max()
x2_min, x2_max = sensor2.min(), sensor2.max()

xx1, xx2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 300),
    np.linspace(x2_min, x2_max, 300)
)

X_grade = np.column_stack((
    np.ones(xx1.ravel().shape[0]),
    xx1.ravel(),
    xx2.ravel()
))


Y_grade_reg = X_grade @ W_reg

classes_grade_reg = np.argmax(Y_grade_reg, axis=1) + 1

classes_grade_reg = classes_grade_reg.reshape(xx1.shape)

plt.figure(figsize=(8, 6))

plt.contourf(
    xx1,
    xx2,
    classes_grade_reg,
    alpha=0.25
)

for classe in range(1, 6):

    indices = classes == classe

    plt.scatter(
        sensor1[indices],
        sensor2[indices],
        label=f"Classe {classe}",
        alpha=0.3
    )

plt.xlabel("Sensor 1 - Corrugador")
plt.ylabel("Sensor 2 - Zigomático")
plt.title(
    f"MQO Regularizado - Acurácia: {accuracy_reg:.4f}"
)
print("Diferença entre W:", np.max(np.abs(W - W_reg)))
plt.legend()
plt.grid()


# MQO Polinomial
X_norm = X / 4095.0

def criar_polinomio(X, q):

    x1 = X[:, 0]
    x2 = X[:, 1]

    caracteristicas = [np.ones(len(X))]

    for grau in range(1, q + 1):

        for i in range(grau + 1):

            j = grau - i

            caracteristicas.append(
                (x1 ** i) * (x2 ** j)
            )

    return np.column_stack(caracteristicas)
resultados = []

print("\n--- MQO POLINOMIAL ---")

for q in range(1, 6):

    inicio = time.perf_counter()
    
    X_poly = criar_polinomio(X_norm, q)
    W_poly = np.linalg.pinv(X_poly) @ Y
    Y_pred_poly = X_poly @ W_poly
    pred_classes_poly = np.argmax(
        Y_pred_poly,
        axis=1
    ) + 1

    
    accuracy_poly = np.mean(
        pred_classes_poly == classes
    )

    fim = time.perf_counter()

    tempo = fim - inicio

    resultados.append(
        [q, accuracy_poly, tempo]
    )

    print(
        f"q={q} | "
        f"Acurácia={accuracy_poly:.5f} | "
        f"Tempo={tempo:.5f}s | "
        f"Características={X_poly.shape[1]}"
    )

    # Mostrar polinomial!!!!!!!!!!!!!!!!!!!

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
axes = axes.ravel()

for indice, q in enumerate(range(2, 6)):

    X_poly = criar_polinomio(X_norm, q)

    W_poly = np.linalg.pinv(X_poly) @ Y

    Y_pred_poly = X_poly @ W_poly
    pred_classes_poly = np.argmax(Y_pred_poly, axis=1) + 1

    accuracy_poly = np.mean(pred_classes_poly == classes)

    x1_min, x1_max = X_norm[:, 0].min(), X_norm[:, 0].max()
    x2_min, x2_max = X_norm[:, 1].min(), X_norm[:, 1].max()

    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 300),
        np.linspace(x2_min, x2_max, 300)
    )

    X_grade = np.column_stack((xx1.ravel(), xx2.ravel()))
    
    X_grade_poly = criar_polinomio(X_grade, q)

    Y_grade = X_grade_poly @ W_poly
    classes_grade = np.argmax(Y_grade, axis=1) + 1
    classes_grade = classes_grade.reshape(xx1.shape)

    
    xx1_original = xx1 * 4095
    xx2_original = xx2 * 4095

    
    ax = axes[indice]

    
    ax.contourf(
        xx1_original,
        xx2_original,
        classes_grade,
        alpha=0.25
    )

   
    for classe in range(1, 6):
        indices = classes == classe

        ax.scatter(
            sensor1[indices],
            sensor2[indices],
            label=f"Classe {classe}",
            alpha=0.5,
            s=8
        )

    ax.set_xlabel("Sensor 1 - Corrugador")
    ax.set_ylabel("Sensor 2 - Zigomático")
    ax.set_title(
        f"MQO Polinomial - q={q}\n"
        f"Acurácia = {accuracy_poly:.4f}"
    )
    ax.grid()

# Espaçamento entre os gráficos
plt.subplots_adjust(
    wspace=0.25,
    hspace=0.35,
    bottom=0.12
)


handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=5
)

plt.show()

