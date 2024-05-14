from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from experiments.metrics.infomredness import infomredness

import time

def  model_builder(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    bi = infomredness(cm)
    return {"model": model, "accuracy": accuracy, "roc": roc, "cm": cm, "bi": bi, "elapsed_time": elapsed_time}

def logistic_regression(X_train, X_test, y_train, y_test, penalty = None):
    solver = "lbfgs"
    if penalty != None and penalty != "l2":
        solver = 'saga'
    model = LogisticRegression(solver=solver,max_iter=1000, penalty=penalty, n_jobs = -1, l1_ratio = 1/2)
    return model_builder(model, X_train, X_test, y_train, y_test)


def knn(X_train, X_test, y_train, y_test, k=1):
    model = KNeighborsClassifier(n_neighbors=k)
    return model_builder(model, X_train, X_test, y_train, y_test)

def naive_bayes(X_train, X_test, y_train, y_test, var_smoothing = 1e-9):
    model = GaussianNB(var_smoothing=var_smoothing)
    return model_builder(model, X_train, X_test, y_train, y_test)

def svm(X_train, X_test, y_train, y_test, C=1, kernel = "rbf"):
    model = SVC(C=C, kernel=kernel)
    return model_builder(model, X_train, X_test, y_train, y_test)

def bagging(X_train, X_test, y_train, y_test, n_estimators):
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=n_estimators,
        n_jobs=-1
    )
    return model_builder(model, X_train, X_test, y_train, y_test)

def boosting(X_train, X_test, y_train, y_test, n_estimators):
    ada_clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=n_estimators,
        algorithm="SAMME.R",
        learning_rate=1.0,
        n_jobs=-1
    )
    return model_builder(model, X_train, X_test, y_train, y_test)