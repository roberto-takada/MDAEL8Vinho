import pandas as pd
import matplotlib.pyplot as plot

def main():
    arquivo_entrada = 'winequality-red-clean.csv'
    nomes_das_colunas = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    valores_independentes = 'sulphates'
    valores_dependentes = 'quality'
    dados_na_memoria = pd.read_csv(arquivo_entrada, names= nomes_das_colunas)

    plot.scatter(dados_na_memoria[valores_independentes], dados_na_memoria[valores_dependentes])
    plot.title("Relação " + valores_independentes + " e Qualidade")
    plot.xlabel( valores_independentes + " no Vinho")
    plot.ylabel("Qualidade do Vinho")
    plot.show()

if __name__ == "__main__":
    main()