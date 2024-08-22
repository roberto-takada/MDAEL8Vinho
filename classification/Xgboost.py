from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn
import xgboost as xgb

def Xgboost(dados):
    X = dados.drop('quality', axis=1)
    y = dados['quality']
    
    y = y.apply(lambda x: 1 if x >= 6 else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f'Acurácia: {accuracy}')
    print(f'Matriz de confusão:\n{conf_matrix}')
    print(f'Relatório de classificação:\n{class_report}')
    sn.heatmap(conf_matrix, cmap='coolwarm', annot=True, linewidth=1, fmt='d')
    plt.show()

