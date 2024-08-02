import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

def main():
    arquivo_entrada = 'winequality-white-clean.csv'
    nomes_das_colunas = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    dados_na_memoria = pd.read_csv(arquivo_entrada, names= nomes_das_colunas)

    desvio_padrao = dados_na_memoria.std()
    print("DESVIO PADRAO: \n")
    print(desvio_padrao)
    print("\n\n")
    variancia = pow(desvio_padrao, 2)
    print("VARIANCIA: \n")
    print(variancia)
    print("\n\n")
    media = dados_na_memoria.mean()
    coeficiente_de_variacao = (desvio_padrao / media) * 100
    print("COEFICIENTE DE VARIAÇÃO: \n")
    print(coeficiente_de_variacao)
    print("\n\n")
    menor_valor = dados_na_memoria.min()
    maior_valor = dados_na_memoria.max()
    amplitude = maior_valor - menor_valor
    print("AMPLITUDE: \n")
    print(amplitude)
    print("\n\n")

    PlotaHistograma(dados_na_memoria)
    intervalos = [3,5,7,9]
    dados_na_memoria['Intervalo de Classe'] = pd.cut(dados_na_memoria['quality'], intervalos)
    tabela_de_frequencia = dados_na_memoria['Intervalo de Classe'].value_counts(sort=False)
    tabela_de_frequencia = tabela_de_frequencia.sort_index()
    tabela_de_frequencia_relativa = tabela_de_frequencia / tabela_de_frequencia.sum()
    tabela_de_frequencia_acumulada = tabela_de_frequencia / tabela_de_frequencia.cumsum()
    tabela_final = pd.DataFrame({'Frequencia': tabela_de_frequencia, 'Frequencia Relativa': tabela_de_frequencia_relativa, 'Frequencia Acumulada': tabela_de_frequencia_acumulada})
    print(tabela_final)

def PlotaHistograma(database):
    plt.figure(figsize=
               (5,3))
    sn.histplot(data=database, x='quality', binwidth=1)
    #sn.histplot(data=database, x='quality', binwidth=2)
    plt.xlabel('Valores')
    plt.ylabel('Frequência')
    plt.title('Histograma')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()