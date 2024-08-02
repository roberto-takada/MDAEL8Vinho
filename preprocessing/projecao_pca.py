import pandas as pd
from sklearn.decomposition import PCA

from preprocessing.print_dados import print_dados
from preprocessing.print_graficos import print_pca_3componentes, print_pca_2componentes


def projecao_pca(dados, valores_independentes, valores_dependentes, opcao):
    x = dados.loc[:, valores_independentes].values
    y = dados.loc[:, [valores_dependentes]].values
    pca = PCA()
    componentes_principais = pca.fit_transform(x)
    print("Variancia por Componente: \n")
    print(pca.explained_variance_ratio_.tolist())
    print("\n\n\n")
    if opcao == 0:
        dataframe_principal = pd.DataFrame(data=componentes_principais[:, 0:2],
                                           columns=['Componente Principal 1', 'Componente Principal 2'])
        dataframe_final = pd.concat([dataframe_principal, dados[[valores_dependentes]]], axis=1)
        print_dados(dataframe_final)
        print_pca_2componentes(dataframe_final, valores_dependentes)
    if opcao == 1:
        dataframe_principal = pd.DataFrame(data=componentes_principais[:, 0:3],
                                           columns=['Componente Principal 1', 'Componente Principal 2',
                                                    'Componente Principal 3'])
        dataframe_final = pd.concat([dataframe_principal, dados[[valores_dependentes]]], axis=1)
        print_dados(dataframe_final)
        print_pca_3componentes(dataframe_final, valores_dependentes)






