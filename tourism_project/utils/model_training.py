import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import joblib

# Set environment variable to allow filesystem tracking backend
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# Set up MLflow tracking
mlflow.set_tracking_uri("file:./mlruns") # Use local file store for MLflow tracking
mlflow.set_experiment("Tourism Package Prediction")

print("Loading processed data...")
# Load the processed data
try:
    X_train = pd.read_csv('tourism_project/model_building/X_train.csv')
    X_test = pd.read_csv('tourism_project/model_building/X_test.csv')
    y_train = pd.read_csv('tourism_project/model_building/y_train.csv').squeeze()
    y_test = pd.read_csv('tourism_project/model_building/y_test.csv').squeeze()
    print("Data loaded successfully.")
except FileNotFoundError as e:
    raise RuntimeError(f"Error loading processed data: {e}. Ensure data_preparation.py was run and saved files correctly.")

# Define models and their parameter grids for GridSearchCV
models = {
    "Logistic Regression": {
        "model": LogisticRegression(random_state=42, solver='liblinear'),
        "params": {
            "C": [0.01, 0.1, 1, 10, 100]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        }
    }
}

best_model_overall = None
best_roc_auc = -1

print("Starting model training and experimentation tracking...")

for model_name, config in models.items():
    with mlflow.start_run(run_name=f"{model_name}_GridSearch") as run:
        print(f"\n--- Training {model_name} ---")
        mlflow.log_param("model_type", model_name)

        classifier = config["model"]
        param_grid = config["params"]

        grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        best_classifier = grid_search.best_estimator_
        best_params = grid_search.best_params_

        mlflow.log_params(best_params)

        y_pred = best_classifier.predict(X_test)
        y_proba = best_classifier.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

        mlflow.log_metrics(metrics)

        print(f"Best parameters for {model_name}: {best_params}")
        print(f"Metrics for {model_name} on test set:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

        # Log model
        mlflow.sklearn.log_model(best_classifier, "model",
                                  registered_model_name=f"{model_name}Classifier")

        # Track the best model overall
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_overall = {
                "model_name": model_name,
                "model_object": best_classifier,
                "metrics": metrics,
                "run_id": run.info.run_id
            }

print("\n--- Model Training and Experimentation Tracking Complete ---")

if best_model_overall:
    print(f"\nBest performing model: {best_model_overall['model_name']}")
    print(f"Best ROC AUC: {best_model_overall['metrics']['roc_auc']:.4f}")
    print(f"MLflow Run ID for best model: {best_model_overall['run_id']}")

    # Register the best model as 'Production Model' in MLflow Model Registry
    # The run_id is needed to reference the artifact for model registration.
    # We need to explicitly get the run_id from the logging run.
    # Let's ensure to register only once for the overall best model.
    with mlflow.start_run(run_name="Register_Best_Model_Overall") as register_run:
        # The model_uri should point to the artifact logged in the best_model_overall['run_id']
        model_uri = f"runs:/{best_model_overall['run_id']}/model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="ProductionTourismClassifier"
        )
        print(f"Best model '{registered_model.name}' version {registered_model.version} registered in MLflow Model Registry.")

    # Save the best model locally (optional, for later use or deployment outside MLflow registry)
    model_save_path = "tourism_project/model_building/best_tourism_classifier.pkl"
    joblib.dump(best_model_overall['model_object'], model_save_path)
    print(f"\nBest model object saved locally to {model_save_path}")

print("\nTo view MLflow UI, run the following in a new terminal in the directory where 'mlruns' folder is created:")
print("mlflow ui --port 5000")
