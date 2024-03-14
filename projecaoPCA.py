import pandas as pd
import matplotlib.pyplot as plot
from sklearn.decomposition import PCA

def main():
    arquivo_entrada = 'winequality-white-normalized.csv'
    nomes_das_colunas = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    valores_independentes = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    valores_dependentes = 'quality'
    dados_na_memoria = pd.read_csv(arquivo_entrada, names= nomes_das_colunas)

    x = dados_na_memoria.loc[:, valores_independentes].values
    y = dados_na_memoria.loc[:, [valores_dependentes]].values

    pca = PCA()
    componentes_principais = pca.fit_transform(x)
    print("Variancia por Componente: \n")
    print(pca.explained_variance_ratio_.tolist())
    print("\n\n\n")

    dataframe_principal = pd.DataFrame(data = componentes_principais[:,0:2], columns = ['Componente Principal 1', 'Componente Principal 2'])
    dataframe_final = pd.concat([dataframe_principal, dados_na_memoria[[valores_dependentes]]], axis = 1)
    MostraInformacoesDoDataFrame(dataframe_final)

    VisualizarProjecaoPCA(dataframe_final, valores_dependentes)

def MostraInformacoesDoDataFrame(dados_na_memoria):
    print('PRIMEIRAS 10 LINHAS DO ARQUIVO \n')
    print(dados_na_memoria.head(10))
    print('\n')

    print('INFORMACOES DOS DADOS: \n')
    print(dados_na_memoria.info())
    print('\n')

    print('DESCRICAO DOS DADOS: \n')
    print(dados_na_memoria.describe())
    print('\n')

def VisualizarProjecaoPCA(dataframe_final, coluna_alvo):
    fig = plot.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Componente Principal 1', fontsize = 15)
    ax.set_ylabel('Componente Principal 2', fontsize = 15)
    ax.set_title('PCA de dois Componentes', fontsize = 20)
    alvos = [3, 4, 5, 6, 7, 8, 9]
    cores = ['#ff0f0f','#0f0fff', '#ff0fff', '#0f0f0f', '#0fe', '#0c0c', '#ffff0f']
    for alvo, cores in zip(alvos, cores):
        indices_a_manter = dataframe_final[coluna_alvo] == alvo
        ax.scatter(dataframe_final.loc[indices_a_manter, 'Componente Principal 1'], dataframe_final.loc[indices_a_manter, 'Componente Principal 2'], c = cores, s = 50)
    ax.legend(alvos)
    ax.grid()
    plot.show()

if __name__ == "__main__":
    main()
