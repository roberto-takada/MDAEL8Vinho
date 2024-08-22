from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.decomposition import PCA

def Kmeans(dados):
    X = dados.drop('quality', axis=1)
    
    pca = PCA()
    projected = pca.fit_transform(X)
    dataframe_principal = pd.DataFrame(data=projected[:, 0:2],columns=['Componente Principal 1', 'Componente Principal 2'])
    kmeans = KMeans(n_clusters=2).fit(projected)
    print(kmeans.inertia_)
    centers = kmeans.cluster_centers_
    score = silhouette_score(X, kmeans.labels_)    
    print("For n_clusters = {}, silhouette score is {})".format(10, score))
    plt.scatter(dataframe_principal['Componente Principal 1'], dataframe_principal['Componente Principal 2'], c=kmeans.labels_)
    plt.show()
