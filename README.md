# Data Science Tools

This repository contains notebooks and scripts exploring various data science tools.

## LazyPredict with MLflow

`lazypredict_mlflow.py` demonstrates how to use **LazyPredict** to quickly train a variety of classification models on the iris dataset. The script logs model accuracies and the result table to **MLflow** for experiment tracking.

### Usage

```bash
pip install lazypredict mlflow scikit-learn pandas
python lazypredict_mlflow.py
```

Running the script will print a table of model performance and create an `mlruns` directory with the MLflow experiment.
