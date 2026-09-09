import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Funcoes implementadas no item 3
# Cada funcao recebe os dados de treino e teste e retorna as classes previstas
from classificacao import (
    classificar_mqo,
    classificar_mqo_regularizado,
    classificar_mqo_polinomial
)


# Configuracoes do trabalho
lambdas = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4,
           0.5, 0.6, 0.7, 0.8, 0.9, 1]
k = 10
R = 500
q_polinomial = 5


# Carrega os dados
raiz = Path(__file__).resolve().parents[1]
dados = np.loadtxt(raiz / "Datasets" / "EMG.csv", delimiter=" ")

X = dados[:, 0:2]
y = dados[:, 2].astype(int)


def acuracia(y_real, y_predito):
    return np.mean(y_real == y_predito)


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

        y_predito = classificar_mqo_regularizado(
            X_treino,
            y_treino,
            X_validacao,
            lamb
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
    "MQO regularizado",
    f"MQO polinomial q={q_polinomial}"
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

    # Cada funcao abaixo pertence a implementacao do item 3
    predicoes = [
        classificar_mqo(X_treino, y_treino, X_teste),
        classificar_mqo_regularizado(
            X_treino,
            y_treino,
            X_teste,
            lambda_ideal
        ),
        classificar_mqo_polinomial(
            X_treino,
            y_treino,
            X_teste,
            q_polinomial
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
plt.xticks(range(1, len(nomes_modelos) + 1), nomes_modelos)
plt.ylabel("Acuracia")
plt.title("Resultados das 500 rodadas de Monte Carlo")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(pasta_resultados / "boxplot_monte_carlo.png", dpi=180)
plt.close()
