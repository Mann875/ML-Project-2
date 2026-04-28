import mlflow
import pandas as pd
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

def train_model(df: pd.DataFrame, target_col: str):
    """
    Trains an XGBoost model and logs with MLflow.

    Args:
        df (pd.DataFrame): Feature Dataset.
        target_col (str): Name of the target column.
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(
        n_estimators = 300,
        learning_rate = 0.1,
        max_depth = 5,
        n_jobs = -1,
        eval_metric = "logloss",
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)

        # Log params, metrics and model
        mlflow.log_params("n_estimators", 300)
        mlflow.log_params("accuracy", acc)
        mlflow.log_params("recall", rec)
        mlflow.xgboost.log_model(model, "model")

        # Log dataset so it shows in ML flow UI
        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print(f"Model Trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}")
        