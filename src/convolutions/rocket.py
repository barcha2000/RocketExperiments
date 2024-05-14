from sktime.transformations.panel.rocket import Rocket, MiniRocket
import mlflow
import pandas as pd
from joblib import dump


def ROCKET(X_train, y_train, X_test, y_test, data_name, path, num_kernels=10000, iter_num = 100):

    for i in range(iter_num):
        with mlflow.start_run(run_name=f'ROCKET_{num_kernels}_{i}'):

            trf = Rocket(num_kernels=num_kernels, random_state=i)
            trf.fit(X_train)

            train_ROCKET = trf.transform(X_train)
            test_ROCKET = trf.transform(X_test)

            if not isinstance(train_ROCKET, pd.DataFrame):
                train_ROCKET = pd.DataFrame(X_train_ROCKET)
            if not isinstance(test_ROCKET, pd.DataFrame):
                test_ROCKET = pd.DataFrame(X_test_ROCKET)

            train_ROCKET['y'] = y_train
            train_ROCKET['id'] = range(len(train_ROCKET))
            test_ROCKET['y'] = y_test
            test_ROCKET['id'] = range(len(test_ROCKET))

            train_path = f'{path}\\data\\rocket\\rocket_{num_kernels}\\{data_name}\\train\\X_train_ROCKET_{i}.csv'
            test_path = f'{path}\\data\\rocket\\rocket_{num_kernels}\\{data_name}\\test\\X_test_ROCKET_{i}.csv'
            train_ROCKET.to_csv(train_path, index=False)
            test_ROCKET.to_csv(test_path, index=False)

            trf_path = f'{path}\\models\\rocket\\rocket_{num_kernels}\\{data_name}\\trf_{i}.joblib'
            dump(trf, trf_path)

            mlflow.log_artifact(train_path, "ROCKET_Features")
            mlflow.log_artifact(test_path, "ROCKET_Features")
            mlflow.log_artifact(trf_path, "Models")

            mlflow.log_param('num_kernels', num_kernels)
            mlflow.log_param('random_state', i)

def MINIROCKET(X_train, y_train, X_test, y_test, data_name, path, num_kernels=10000, iter_num = 100):

    for i in range(iter_num):
        with mlflow.start_run(run_name=f'MINIROCKET_{num_kernels}_{i}'):

            trf = MiniRocket(num_kernels=num_kernels, random_state=i)
            trf.fit(X_train)

            train_ROCKET = trf.transform(X_train)
            test_ROCKET = trf.transform(X_test)

            if not isinstance(train_ROCKET, pd.DataFrame):
                train_ROCKET = pd.DataFrame(X_train_ROCKET)
            if not isinstance(test_ROCKET, pd.DataFrame):
                test_ROCKET = pd.DataFrame(X_test_ROCKET)

            train_ROCKET['y'] = y_train
            train_ROCKET['id'] = range(len(train_ROCKET))
            test_ROCKET['y'] = y_test
            test_ROCKET['id'] = range(len(test_ROCKET))

            train_path = f'{path}\\data\\minirocket\\minirocket_{num_kernels}\\{data_name}\\train\\X_train_ROCKET_{i}.csv'
            test_path = f'{path}\\data\\minirocket\\minirocket_{num_kernels}\\{data_name}\\test\\X_test_ROCKET_{i}.csv'
            train_ROCKET.to_csv(train_path, index=False)
            test_ROCKET.to_csv(test_path, index=False)

            trf_path = f'{path}\\models\\minirocket\\minirocket_{num_kernels}\\{data_name}\\trf_{i}.joblib'
            dump(trf, trf_path)

            mlflow.log_artifact(train_path, "MINIROCKET_Features")
            mlflow.log_artifact(test_path, "MINIROCKET_Features")
            mlflow.log_artifact(trf_path, "Models")

            mlflow.log_param('num_kernels', num_kernels)
            mlflow.log_param('random_state', i)