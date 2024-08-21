from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree


def random_forest(dados):
    X = dados.drop('quality', axis=1)
    y = dados['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier(random_state=0)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
