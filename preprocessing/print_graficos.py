import matplotlib.pyplot as plt
import seaborn as sn


def print_grafico_histograma(dados, opcao):
    plt.figure(figsize=
               (5, 3))
    sn.histplot(data=dados, x='quality', binwidth=opcao)
    plt.xlabel('Valores')
    plt.ylabel('Frequência')
    plt.title('Histograma')
    plt.grid(True)
    plt.show()


def print_grafico_caixas(dados, variavel_1, variavel_2):
    sn.boxplot(x=dados[variavel_1], y=dados[variavel_2], width=0.3)
    plt.show()


def print_matriz_correlacao(dados):
    correlacao = dados.corr()
    sn.heatmap(correlacao, annot=True, fmt=".1f", linewidths=.6)
    plt.show()


def print_grafico_2variaveis(dados, variavel1, variavel2):
    plt.scatter(dados[variavel1], dados[variavel2])
    plt.title("Relação " + variavel1 + " e " + variavel2)
    plt.xlabel(variavel1 + " do Vinho")
    plt.ylabel(variavel2 + " do Vinho")
    plt.show()


def print_pca_2componentes(dados, coluna_alvo):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Componente Principal 1', fontsize=15)
    ax.set_ylabel('Componente Principal 2', fontsize=15)
    ax.set_title('PCA de dois Componentes', fontsize=20)
    alvos = [3, 4, 5, 6, 7, 8, 9]
    cores = ['#ff0f0f', '#0f0fff', '#ff0fff', '#0f0f0f', '#0fe', '#0c0c', '#ffff0f']
    for alvo, cores in zip(alvos, cores):
        indices_a_manter = dados[coluna_alvo] == alvo
        ax.scatter(dados.loc[indices_a_manter, 'Componente Principal 1'],
                   dados.loc[indices_a_manter, 'Componente Principal 2'], c=cores, s=50)
    ax.legend(alvos)
    ax.grid()
    plt.show()


def print_pca_3componentes(dados, coluna_alvo):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('Componente Principal 1', fontsize=15)
    ax.set_ylabel('Componente Principal 2', fontsize=15)
    ax.set_zlabel('Componente Principal 3', fontsize=15)
    ax.set_title('PCA de três Componentes', fontsize=20)
    alvos = [3, 4, 5, 6, 7, 8, 9]
    cores = ['#ff0f0f', '#0f0fff', '#ff0fff', '#0f0f0f', '#0fe', '#0c0c', '#ffff0f']
    for alvo, cores in zip(alvos, cores):
        indices_a_manter = dados[coluna_alvo] == alvo
        ax.scatter(dados.loc[indices_a_manter, 'Componente Principal 1'],
                   dados.loc[indices_a_manter, 'Componente Principal 2'],
                   dados.loc[indices_a_manter, 'Componente Principal 3'], c=cores, s=50)
    ax.legend(alvos)
    ax.grid()
    plt.show()
