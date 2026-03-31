# src/models/train.py

""" Training module for the movie recommendation system. 
This module contains functions to train the recommendation model using the training data."""

import joblib
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

from src.utils import get_logger

logger = get_logger(__name__)

# Define the path to save the trained model
MODELS_PATH = Path(__file__).parent.parent.parent / "models"

def train_model(user_item_matrix: csr_matrix, k: int = 50) -> NearestNeighbors:
    """
    Train ItemKNN model on user-item matrix.
    
    Args:
        user_item_matrix: sparse user-item matrix
        k: number of neighbors
    Returns:
        trained NearestNeighbors model
    """
    logger.info(f"Training ItemKNN with K={k}")
    
    model = NearestNeighbors(
        n_neighbors=k,
        metric="cosine",
        algorithm="brute"
    )
    
    model.fit(user_item_matrix.T)
    
    logger.info(f"Model trained successfully!")
    
    return model

def save_model(model: NearestNeighbors, model_name: str = "itemknn") -> None:
    """
    Save trained model to models/ folder.
    
    Args:
        model: trained NearestNeighbors model
        model_name: name of the model file
    """
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    
    model_path = MODELS_PATH / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    
    logger.info(f"Model saved to {model_path} successfully!")

def _evaluate_model(
    model: NearestNeighbors,
    user_item_matrix: csr_matrix,
    test: pd.DataFrame,
    user_map: dict,
    item_map: dict
) -> float:
    """
    Evaluate model using RMSE on test data.
    
    Args:
        model: trained model
        user_item_matrix: sparse matrix
        test: test dataframe
        user_map: user_id to index mapping
        item_map: movie_id to index mapping
    Returns:
        RMSE score
    """
    from sklearn.metrics import root_mean_squared_error

    actuals = []
    predictions = []

    for _, row in test.head(1000).iterrows():
        user_idx = user_map.get(row["user_id"])
        item_idx = item_map.get(row["movie_id"])

        if user_idx is None or item_idx is None:
            continue

        distances, indices = model.kneighbors(
            user_item_matrix.T[item_idx],
            n_neighbors=model.n_neighbors
        )

        similar_ratings = user_item_matrix[user_idx, indices[0]].toarray().flatten()
        weights = 1 - distances[0]

        if weights.sum() > 0 and similar_ratings.sum() > 0:
            pred = np.average(similar_ratings, weights=weights)
        else:
            pred = 0

        actuals.append(row["rating"])
        predictions.append(pred)

    rmse = root_mean_squared_error(actuals, predictions)
    logger.info(f"RMSE: {rmse:.4f}")
    return rmse

def train_pipeline(
    user_item_matrix: csr_matrix,
    test,
    user_map: dict,
    item_map: dict,
    k_values: list = [10, 20, 50]
) -> NearestNeighbors:
    """
    Train multiple ItemKNN models with different K values,
    track with MLflow and save the best model.
    
    Args:
        user_item_matrix: sparse user-item matrix
        test: test dataframe
        user_map: user_id to index mapping
        item_map: movie_id to index mapping
        k_values: list of K values to try
    Returns:
        best trained model
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("movie-recommendation-system")

    best_model = None
    best_k = None
    best_rmse = float("inf")

    for k in k_values:
        with mlflow.start_run(run_name=f"ItemKNN_K{k}"):

            # train model
            model = train_model(user_item_matrix, k=k)

            # evaluate model
            rmse = _evaluate_model(model, user_item_matrix, test, user_map, item_map)

            # log to mlflow
            mlflow.log_param("k", k)
            mlflow.log_param("metric", "cosine")
            mlflow.log_param("algorithm", "brute")
            mlflow.log_metric("rmse", rmse)

            logger.info(f"K={k} → RMSE: {rmse:.4f}")

            # track best model
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_k = k

    # save best model
    save_model(best_model, model_name=f"itemknn_k{best_k}")

    logger.info(f"Best model: ItemKNN K={best_k} RMSE={best_rmse:.4f} ✅")

    return best_model