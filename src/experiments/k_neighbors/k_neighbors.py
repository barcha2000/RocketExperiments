import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import pandas as pd

from src import data_transform, knn




def knn_experiment(path, dataset_name, algortihm_type, iter_num, metric_type = "", k = 1):

    for i in range(iter_num):
        (X_train, X_test, y_train, y_test) = data_transform(path, metric_type, i)

        run_name = f"KNN_{dataset_name}_{algortihm_type}_{k}_{metric_type}_{i}"

        results = knn(X_train, X_test, y_train, y_test, k=k)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({"k": k, "n_neighbors": k})
            mlflow.log_metrics({"accuracy": results["accuracy"], "roc": results["roc"], "bi": results["bi"],
                                "elapsed_time": results["elapsed_time"]})
            mlflow.sklearn.log_model(results["model"], f"model_k_{k}")

            cm_path = f"C:\\Users\\barcha\\tsVenv\\data\\knn\\{dataset_name}_{algortihm_type}_{metric_type}\\confusion_matrix_{i}.csv"
            pd.DataFrame(results['cm']).to_csv(cm_path, index=False)
            mlflow.log_artifact(cm_path, "confusion_matrices")