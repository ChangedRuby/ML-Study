import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Configuracoes do trabalho
lambdas = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
k = 10
R = 500


# Carrega os dados
raiz = Path(__file__).resolve().parents[1]
dados = np.loadtxt(raiz / "Datasets" / "EMG.csv", delimiter=" ")

X = dados[:, 0:2]
y = dados[:, 2].astype(int)
classes = np.unique(y)


def acuracia(y_real, y_predito):
    return np.mean(y_real == y_predito)


# MQO multiclasse
def treinar_mqo(X_treino, y_treino):
    X_com_1 = np.column_stack((np.ones(len(X_treino)), X_treino))

    # A classe correta recebe 1 e as outras recebem -1
    Y = -np.ones((len(y_treino), len(classes)))
    for i, classe in enumerate(classes):
        Y[y_treino == classe, i] = 1

    W = np.linalg.pinv(X_com_1) @ Y
    return W


def predizer_mqo(X_teste, W):
    X_com_1 = np.column_stack((np.ones(len(X_teste)), X_teste))
    pontuacoes = X_com_1 @ W
    return classes[np.argmax(pontuacoes, axis=1)]


# Calcula os parametros usados pelos classificadores gaussianos
def treinar_gaussiano(X_treino, y_treino):
    medias = []
    covariancias = []
    probabilidades = []
    quantidades = []

    for classe in classes:
        X_classe = X_treino[y_treino == classe]

        medias.append(np.mean(X_classe, axis=0))
        covariancias.append(np.cov(X_classe.T))
        probabilidades.append(len(X_classe) / len(X_treino))
        quantidades.append(len(X_classe))

    medias = np.array(medias)
    covariancias = np.array(covariancias)
    probabilidades = np.array(probabilidades)
    quantidades = np.array(quantidades)

    # Covariancia calculada com todos os dados de treinamento
    covariancia_total = np.cov(X_treino.T)

    # Media ponderada das covariancias das classes
    covariancia_agregada = np.zeros((2, 2))
    for i in range(len(classes)):
        covariancia_agregada += (quantidades[i] - 1) * covariancias[i]

    covariancia_agregada /= len(X_treino) - len(classes)

    return medias, covariancias, probabilidades, covariancia_total, covariancia_agregada


def predizer_gaussiano(X_teste, parametros, tipo, lamb=0):
    medias, covariancias, probabilidades, cov_total, cov_agregada = parametros
    pontuacoes = np.zeros((len(X_teste), len(classes)))

    for i in range(len(classes)):
        if tipo == "tradicional":
            cov = covariancias[i]

        elif tipo == "total":
            cov = cov_total

        elif tipo == "agregada":
            cov = cov_agregada

        elif tipo == "naive":
            # No Naive Bayes, as covariancias fora da diagonal viram zero
            cov = np.diag(np.diag(covariancias[i]))

        elif tipo == "regularizado":
            cov = (1 - lamb) * covariancias[i] + lamb * cov_agregada

        # Evita erro caso a matriz seja singular
        cov = cov + 1e-6 * np.eye(2)

        diferenca = X_teste - medias[i]
        inversa = np.linalg.pinv(cov)

        # Distancia de Mahalanobis de cada amostra ate a classe
        distancia = np.sum((diferenca @ inversa) * diferenca, axis=1)

        # Log da probabilidade gaussiana
        pontuacoes[:, i] = (
            -0.5 * np.log(np.linalg.det(cov))
            -0.5 * distancia
            + np.log(probabilidades[i])
        )

    return classes[np.argmax(pontuacoes, axis=1)]


# ============================================================
# TOPICO 4 - K-FOLD PARA ESCOLHER O MELHOR LAMBDA
# ============================================================

np.random.seed(42)
indices = np.random.permutation(len(X))
folds = np.array_split(indices, k)

medias_lambda = []
desvios_lambda = []

for lamb in lambdas:
    acuracias_lambda = []

    for i in range(k):
        indices_validacao = folds[i]
        indices_treino = np.concatenate(folds[:i] + folds[i + 1:])

        X_treino = X[indices_treino]
        y_treino = y[indices_treino]
        X_validacao = X[indices_validacao]
        y_validacao = y[indices_validacao]

        parametros = treinar_gaussiano(X_treino, y_treino)
        y_predito = predizer_gaussiano(
            X_validacao, parametros, "regularizado", lamb
        )

        acuracias_lambda.append(acuracia(y_validacao, y_predito))

    medias_lambda.append(np.mean(acuracias_lambda))
    desvios_lambda.append(np.std(acuracias_lambda))

melhor_indice = np.argmax(medias_lambda)
lambda_ideal = lambdas[melhor_indice]

print("\nResultado do k-fold:")
for lamb, media in zip(lambdas, medias_lambda):
    print(f"lambda = {lamb:<5} acuracia media = {media:.5f}")

print(f"\nMelhor lambda: {lambda_ideal}")


# ============================================================
# TOPICO 5 - VALIDACAO MONTE CARLO COM 500 RODADAS
# ============================================================

nomes_modelos = [
    "MQO tradicional",
    "Gaussiano tradicional",
    "Gaussiano cov. total",
    "Gaussiano cov. agregada",
    "Bayes ingenuo",
    "Gaussiano regularizado"
]

# Cada modelo tera uma lista com 500 acuracias
resultados = {nome: [] for nome in nomes_modelos}

np.random.seed(43)

for rodada in range(R):
    indices = np.random.permutation(len(X))
    corte = int(0.8 * len(X))

    indices_treino = indices[:corte]
    indices_teste = indices[corte:]

    X_treino = X[indices_treino]
    y_treino = y[indices_treino]
    X_teste = X[indices_teste]
    y_teste = y[indices_teste]

    # Treina os modelos
    W = treinar_mqo(X_treino, y_treino)
    parametros = treinar_gaussiano(X_treino, y_treino)

    # Faz as predicoes
    predicoes = [
        predizer_mqo(X_teste, W),
        predizer_gaussiano(X_teste, parametros, "tradicional"),
        predizer_gaussiano(X_teste, parametros, "total"),
        predizer_gaussiano(X_teste, parametros, "agregada"),
        predizer_gaussiano(X_teste, parametros, "naive"),
        predizer_gaussiano(
            X_teste, parametros, "regularizado", lambda_ideal
        )
    ]

    # Guarda a acuracia de cada modelo
    for nome, y_predito in zip(nomes_modelos, predicoes):
        resultados[nome].append(acuracia(y_teste, y_predito))

    if (rodada + 1) % 50 == 0:
        print(f"Rodada {rodada + 1}/{R}")


# ============================================================
# TOPICO 6 - TABELA COM OS RESULTADOS
# ============================================================

resumo = []

for nome in nomes_modelos:
    valores = resultados[nome]

    resumo.append([
        nome,
        np.mean(valores),
        np.std(valores),
        np.max(valores),
        np.min(valores)
    ])

print("\n")
print(f"{'Modelo':<30} {'Media':>9} {'Desvio':>9} {'Maior':>9} {'Menor':>9}")
print("-" * 70)

for nome, media, desvio, maior, menor in resumo:
    print(f"{nome:<30} {media:>9.5f} {desvio:>9.5f} "
          f"{maior:>9.5f} {menor:>9.5f}")


# ============================================================
# SALVA AS TABELAS E OS GRAFICOS
# ============================================================

pasta_resultados = raiz / "Resultados"
pasta_resultados.mkdir(exist_ok=True)

# Salva os resultados do k-fold
np.savetxt(
    pasta_resultados / "validacao_lambda.csv",
    np.column_stack((lambdas, medias_lambda, desvios_lambda)),
    delimiter=",",
    header="lambda,acuracia_media,desvio_padrao",
    comments="",
    fmt="%.8f"
)

# Salva a tabela do Monte Carlo
with open(pasta_resultados / "resumo_monte_carlo.csv", "w", encoding="utf-8") as arquivo:
    arquivo.write("modelo,media,desvio_padrao,maior_valor,menor_valor\n")

    for nome, media, desvio, maior, menor in resumo:
        arquivo.write(
            f'"{nome}",{media:.8f},{desvio:.8f},{maior:.8f},{menor:.8f}\n'
        )

# Grafico dos lambdas
plt.figure(figsize=(9, 5))
posicoes = np.arange(len(lambdas))
plt.plot(posicoes, medias_lambda, marker="o")
plt.xticks(posicoes, lambdas)
plt.xlabel("Lambda")
plt.ylabel("Acuracia media")
plt.title("Escolha do lambda por k-fold")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(pasta_resultados / "validacao_lambda.png", dpi=180)
plt.close()

# Boxplot das 500 acuracias de cada modelo
plt.figure(figsize=(11, 6))
plt.boxplot([resultados[nome] for nome in nomes_modelos])
plt.xticks(range(1, 7), nomes_modelos, rotation=20, ha="right")
plt.ylabel("Acuracia")
plt.title("Resultados das 500 rodadas de Monte Carlo")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(pasta_resultados / "boxplot_monte_carlo.png", dpi=180)
plt.close()
