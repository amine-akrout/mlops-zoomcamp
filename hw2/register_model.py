import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from hyperopt.pyll import scope
from hyperopt import hp, space_eval

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = {
    "max_depth": scope.int(hp.quniform("max_depth", 1, 20, 1)),
    "n_estimators": scope.int(hp.quniform("n_estimators", 10, 50, 1)),
    "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 10, 1)),
    "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 4, 1)),
    "random_state": 42,
}

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        params = space_eval(RF_PARAMS, params)
        rf = RandomForestRegressor(**params)
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        # Evaluate model on the validation and test sets
        val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
        mlflow.log_metric("val_rmse", float(val_rmse))
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", float(test_rmse))


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved",
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote",
)
def run_register_model(data_path: str, top_n: int):
    client = MlflowClient()

    # Create the experiment if it does not exist
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(HPO_EXPERIMENT_NAME)
        experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)

    # Retrieve the top_n model runs and log the models
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"],
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    # get best run
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"],
    )[0]

    # Register the best model
    mlflow.register_model(
        f"runs:/{best_run.info.run_id}/model", "best-random-forest-model"
    )


if __name__ == "__main__":
    run_register_model()
