import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
import mlflow


def main():
    data = load_iris(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    # save results for reference
    models.to_csv("lazypredict_results.csv")
    print(models)

    mlflow.set_experiment("lazypredict_iris")
    with mlflow.start_run():
        mlflow.log_param("dataset", "iris")
        for model_name, row in models.iterrows():
            mlflow.log_metric(f"{model_name}_accuracy", row["Accuracy"])
        mlflow.log_artifact("lazypredict_results.csv")


if __name__ == "__main__":
    main()
