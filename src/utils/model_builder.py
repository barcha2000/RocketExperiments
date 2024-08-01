from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from src import infomredness


def  model_builder(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    bi = infomredness(cm)
    return {"model": model, "accuracy": accuracy, "roc": roc, "cm": cm, "bi": bi}
