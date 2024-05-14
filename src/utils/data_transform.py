import pandas as pd

def data_transform(path, metric_type, i):

    train_path = path + f"/train/X_train_ROCKET_{i}.csv"
    test_path = path + f"/test/X_test_ROCKET_{i}.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.drop(['y', 'id'], axis=1)
    y_train = train_data['y']
    X_test = test_data.drop(['y', 'id'], axis=1)
    y_test = test_data['y']

    if metric_type == "maximum":
        X_train = X_train.iloc[:, 1::2]
        X_test = X_test.iloc[:, 1::2]

    if metric_type == "ppv":
        X_train = X_train.iloc[:, ::2]
        X_test = X_test.iloc[:, ::2]

    return X_train, X_test, y_train, y_test
