from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sn

def decision_tree(dados):
    X = dados.drop('quality', axis=1)
    y = dados['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = tree.DecisionTreeClassifier(random_state=0)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #tree.plot_tree(clf)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    sn.heatmap(conf_matrix, cmap='coolwarm', annot=True, linewidth=1, fmt='d')
    plt.show()
    class_report = metrics.classification_report(y_test, y_pred)
    print(f'Relatório de classificação:\n{class_report}')
    


