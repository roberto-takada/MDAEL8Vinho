import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def main():
    arquivo_entrada = 'winequality-white-clean.csv'
    nomes_das_colunas = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    #valores_dependentes = 'quality'
    dados_na_memoria = pd.read_csv(arquivo_entrada, names= nomes_das_colunas)
    correlation=dados_na_memoria.corr()
    PlotMatrizCorr(correlation)

    

def PlotMatrizCorr(correlacao):
    plot = sn.heatmap(correlacao, annot = True, fmt=".1f", linewidths=.6)
    plt.show()

if __name__ == "__main__":
    main()