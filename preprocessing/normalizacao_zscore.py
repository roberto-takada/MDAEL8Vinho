from sklearn.preprocessing import StandardScaler
import pandas as pd

from preprocessing.print_dados import print_dados


def z_score(dados, valores_independentes, valores_dependentes, arquivo_saida):


    x = dados.loc[:, valores_independentes].values
    y = dados.loc[:, [valores_dependentes]].values

    x_zscore = StandardScaler().fit_transform(x)
    dataframe_normalize_zscore = pd.DataFrame(data=x_zscore, columns=valores_independentes)
    dataframe_variaveis_dependentes = pd.DataFrame(data=y, columns=[valores_dependentes])
    dataframe_normalize_zscore = pd.concat([dataframe_normalize_zscore, dataframe_variaveis_dependentes], axis=1)
    print_dados(dataframe_normalize_zscore)
    dataframe_normalize_zscore.to_csv(arquivo_saida, header=False, index=False)
