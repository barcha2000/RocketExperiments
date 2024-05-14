from sktime.datasets import load_UCR_UEA_dataset

from itertools import product

from src import ROCKET, MINIROCKET

import mlflow
from mlflow.tracking import MlflowClient




# data_name = "ECG200"
# path = "C:\\Users\\barcha\\tsVenv"
#
# X_train, y_train = load_UCR_UEA_dataset(data_name, split="test", return_X_y=True)
# X_test, y_test = load_UCR_UEA_dataset(data_name, split="train", return_X_y=True)
#
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

data_name = "MelbournePedestrian"
path = "C:\\Users\\barcha\\tsVenv"

X_train, y_train = load_UCR_UEA_dataset(data_name, split="test", return_X_y=True)
X_test, y_test = load_UCR_UEA_dataset(data_name, split="train", return_X_y=True)



"""
Rocket
"""

ROCKET_params_grid = {"num_kernels": [250, 500, 1000, 2500, 5000, 10000], "iter_num": [100]}
# ROCKET_params_grid = {"num_kernels": [1000], "iter_num": [100]}

all_combinations_ROCKET = [dict(zip(ROCKET_params_grid.keys(), values)) for values in product(*ROCKET_params_grid.values())]


def run_experiment_ROCKET():
    mlflow.set_experiment('ROCKET_ECG200')
    for comb in all_combinations_ROCKET:
        ROCKET(X_train, y_train, X_test, y_test, data_name, path, **comb)

def run_experiment_MINIROCKET():
    mlflow.set_experiment('MINIROCKET_ECG200')
    for comb in all_combinations_ROCKET:
        MINIROCKET(X_train, y_train, X_test, y_test, data_name, path, **comb)

if __name__ == '__main__':
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    run_experiment_MINIROCKET()