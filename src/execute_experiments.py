from itertools import product

import mlflow

from experiments.logistic_regression.logistic_regression_experiment import logistic_regression_experiment
from experiments.k_neighbors.k_neighbors import knn_experiment
from experiments.naive_bayes.naive_bayes import naive_bayes_experiment
from experiments.svm.svm import svm_experiment
from experiments.bagging.bagging import bagging_experiment
from experiments.boosting.boosting import boosting_experiment

from google.cloud import storage

import logging
import os
import sys
import shutil

def run_experiment(experiment, datasets, params_grid):
    all_combinations_params_grid = [dict(zip(params_grid.keys(), values)) for values in
                               product(*params_grid.values())]

    for comb in all_combinations_params_grid:
        logger.info("Experiment started for params: {}".format(comb))
        experiment(**datasets, **comb)

def download_folder(bucket_name, folder_name, destination_directory):
    """Pobiera wszystkie pliki z folderu w kubie Google Cloud Storage do lokalnego katalogu."""
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=folder_name)

    if not os.path.exists(destination_directory + "/" + "train"):
        os.makedirs(destination_directory + "/" + "train")
        logger.info("Created folder " + destination_directory + "/" + "train")

    if not os.path.exists(destination_directory + "/" + "test"):
        os.makedirs(destination_directory + "/" + "test")
        logger.info("Created folder " + destination_directory + "/" + "test")

    for blob in blobs:
        destination_file_name = os.path.join(destination_directory, os.path.relpath(blob.name, folder_name))
        blob.download_to_filename(destination_file_name)
    logger.info("Downloaded folder: {}".format(folder_name))

datasets = [
    {'path': "../data/rocket/rocket_250/ECG200", 'dataset_name': "ECG200", "algortihm_type": "rocket_250"},
    {'path': "../data/rocket/rocket_500/ECG200", 'dataset_name': "ECG200", "algortihm_type": "rocket_500"},
    {'path': "../data/rocket/rocket_1000/ECG200", 'dataset_name': "ECG200", "algortihm_type": "rocket_1000"},
    {'path': "../data/rocket/rocket_2500/ECG200", 'dataset_name': "ECG200", "algortihm_type": "rocket_2500"},
    {'path': "../data/rocket/rocket_5000/ECG200", 'dataset_name': "ECG200", "algortihm_type": "rocket_5000"},
    {'path': "../data/rocket/rocket_10000/ECG200", 'dataset_name': "ECG200", "algortihm_type": "rocket_10000"},
    {'path': "../data/minirocket/minirocket_250/ECG200", 'dataset_name': "ECG200", "algortihm_type": "minirocket_250"},
    {'path': "../data/minirocket/minirocket_500/ECG200", 'dataset_name': "ECG200", "algortihm_type": "minirocket_500"},
    {'path': "../data/minirocket/minirocket_1000/ECG200", 'dataset_name': "ECG200", "algortihm_type": "minirocket_1000"},
    {'path': "../data/minirocket/minirocket_2500/ECG200", 'dataset_name': "ECG200", "algortihm_type": "minirocket_2500"},
    {'path': "../data/minirocket/minirocket_5000/ECG200", 'dataset_name': "ECG200", "algortihm_type": "minirocket_5000"},
    {'path': "../data/minirocket/minirocket_10000/ECG200", 'dataset_name': "ECG200", "algortihm_type": "minirocket_10000"}
]


"""
Logistic regression
"""
params_grid = {
    'iter_num': [100],
    # 'metric_type': ["", "ppv", "max"],
    'metric_type': ["ppv", "max"],
    'penalty': ['l1', 'l2', 'elasticnet', None]
}

if __name__ == '__main__':
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment('LOGISTIC_REGRESSION_ECG200')
    for i in datasets:
        run_experiment(logistic_regression_experiment,i, params_grid)




"""
K-Neighbours
"""


params_grid = {
    'iter_num': [100],
    'metric_type': ["", "ppv", "max"],
    'k': [i for i in range(1,11)]
}

if __name__ == '__main__':
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment('KNN_ECG200')
    for i in datasets:
        run_experiment(knn_experiment,i, params_grid)


"""
Naive Bayes
"""


params_grid = {
    'iter_num': [100],
    'metric_type': ["", "ppv", "max"],
    'k': [1e-10,1e-9, 1e-8]
}

if __name__ == '__main__':
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment('NAIVEBAYES_ECG200')
    for i in datasets:
        run_experiment(naive_bayes_experiment,i, params_grid)


"""
SVM
"""


params_grid = {
    'iter_num': [100],
    # 'metric_type': ["max"],
    'metric_type': ["","ppv", "max"],
    'c': [10**i for i in range(2, 5)],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
    # 'kernel': ['linear', 'poly', 'sigmoid']
    # 'kernel': ['poly', 'sigmoid']

}

if __name__ == '__main__':
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment('SVM_ECG200')
    for i in datasets:
        run_experiment(svm_experiment,i, params_grid)


mlflow server    --backend-store-uri sqlite:///mlflow.db   --default-artifact-root gs://rocket_experiments/mlflow

"""
BAGGING
"""


params_grid = {
    'iter_num': [100],
    'metric_type': ["", "ppv", "max"],
    'k': [80, 90, 100]
}

if __name__ == '__main__':
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parent_directory = os.path.abspath(os.path.join(sys.path[0], os.pardir))
    logs_directory = os.path.join(parent_directory, 'logs')
    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)
    logger = logging.getLogger(__name__)
    log_file_path = os.path.join(logs_directory, 'experiments.log')
    file_handler = logging.FileHandler(log_file_path)
    logger.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    try:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment('BAGGING_ECG200')
        for i in datasets:
            download_folder("rocket_experiments", i["path"][3:], i["path"])
            logger.info("Starting experiment for dataset: {}".format(i))
            run_experiment(bagging_experiment,i, params_grid)
            shutil.rmtree(i["path"])
    except Exception as e:
        logger.exception(e)

"""
BOOSTING
"""


params_grid = {
    'iter_num': [100],
    'metric_type': ["", "ppv", "max"],
    'k': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
}

if __name__ == '__main__':
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment('BAGGING_ECG200')
    for i in datasets:
        run_experiment(boosting_experiment,i, params_grid)
