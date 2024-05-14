import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import pandas as pd

from src import data_transform, logistic_regression



def logistic_regression_experiment(path, dataset_name, algortihm_type, iter_num, metric_type = "", penalty=None):
    penalty_name = ""
    if penalty != None:
        penalty_name = penalty
    else:
        penalty_name = ""

    for i in range(iter_num):

        run_name = f"LogReg_{dataset_name}_{algortihm_type}_{penalty_name}_{metric_type}_{i}"

        (X_train, X_test, y_train, y_test) = data_transform(path, metric_type, i)

        results = logistic_regression(X_train, X_test, y_train, y_test, penalty=penalty)


        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({"solver": "lbfgs", "max_iter": 1000, "penalty": penalty, "dataset_name":dataset_name, "algortihm_type":algortihm_type, "metric_type" : metric_type})
            mlflow.log_metrics({"accuracy": results["accuracy"], "roc": results["roc"], "bi": results["bi"], "elapsed_time": results["elapsed_time"]})
            mlflow.sklearn.log_model(results["model"], "model")

            cm_path = f"C:\\Users\\barcha\\tsVenv\\data\\logreg\\{dataset_name}_{algortihm_type}_{metric_type}\\confusion_matrix_{penalty}_{i}.csv"
            pd.DataFrame(results['cm']).to_csv(cm_path, index=False)
            mlflow.log_artifact(cm_path, "confusion_matrices")

            mlflow.sklearn.log_model(results['model'], f"LogReg_ROCKET_{penalty_name}_{i}.joblib")


