import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

def main():
    arquivo_entrada = 'winequality-all.csv'
    nomes_das_colunas = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality', 'type']
    dados_na_memoria = pd.read_csv(arquivo_entrada, names= nomes_das_colunas)

    sn.boxplot(x=dados_na_memoria['type'], y = dados_na_memoria['residual sugar'], width = 0.3)
    plt.show()

if __name__ == "__main__":
    main()