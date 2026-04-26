# src/models/train.py

"""Training module for the movie recommendation system."""

import joblib
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

from src.utils import get_logger

logger = get_logger(__name__)

MODELS_PATH = Path(__file__).parent.parent.parent / "models"


def train_model(user_item_matrix: csr_matrix, k: int = 50) -> NearestNeighbors:
    """
    Train ItemKNN model on normalized user-item matrix.

    Args:
        user_item_matrix: normalized sparse matrix
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


def train_svd_model(normalized_matrix: csr_matrix, n_factors: int = 50) -> tuple:
    """
    Train SVD model on normalized user-item matrix.

    Args:
        normalized_matrix: normalized sparse matrix
        n_factors: number of latent factors
    Returns:
        tuple: U, sigma, Vt, predicted_ratings
    """
    from scipy.sparse.linalg import svds

    logger.info(f"Training SVD with {n_factors} factors")

    U, sigma, Vt = svds(normalized_matrix.astype(float), k=n_factors)
    sigma = np.diag(sigma)

    predicted_ratings = np.dot(np.dot(U, sigma), Vt)

    if hasattr(predicted_ratings, "toarray"):
        predicted_ratings = predicted_ratings.toarray()

    predicted_ratings = np.asarray(predicted_ratings)

    logger.info(f"SVD trained successfully!")
    return U, sigma, Vt, predicted_ratings


def save_model(model, model_name: str = "itemknn") -> None:
    """
    Save trained model to models/ folder.

    Args:
        model: trained model or predicted ratings matrix
        model_name: name of the model file
    """
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_PATH / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path} successfully!")


def _evaluate_itemknn(
    model: NearestNeighbors,
    user_item_matrix: csr_matrix,
    normalized_matrix: csr_matrix,
    user_means: np.ndarray,
    test: pd.DataFrame,
    user_map: dict,
    item_map: dict
) -> float:
    """
    Evaluate ItemKNN model using RMSE on test sample.

    Args:
        model: trained ItemKNN model
        user_item_matrix: original sparse matrix
        normalized_matrix: normalized sparse matrix
        user_means: array of user means
        test: test dataframe
        user_map: user_id to index mapping
        item_map: movie_id to index mapping
    Returns:
        RMSE score
    """
    from sklearn.metrics import root_mean_squared_error

    actuals = []
    predictions = []

    test_sample = test.sample(1000, random_state=42)

    for _, row in test_sample.iterrows():
        user_idx = user_map.get(row["user_id"])
        item_idx = item_map.get(row["movie_id"])

        if user_idx is None or item_idx is None:
            continue

        distances, indices = model.kneighbors(
            normalized_matrix.T[item_idx],
            n_neighbors=model.n_neighbors
        )

        similar_ratings = user_item_matrix[user_idx, indices[0]].toarray().flatten()
        weights = 1 - distances[0]

        if weights.sum() > 0 and similar_ratings.sum() > 0:
            pred = np.average(similar_ratings, weights=weights)
        else:
            pred = user_means[user_idx]

        actuals.append(row["rating"])
        predictions.append(pred)

    rmse = root_mean_squared_error(actuals, predictions)
    logger.info(f"ItemKNN RMSE: {rmse:.4f}")
    return rmse


def _evaluate_svd(
    predicted_ratings: np.ndarray,
    user_means: np.ndarray,
    test: pd.DataFrame,
    user_map: dict,
    item_map: dict
) -> float:
    """
    Evaluate SVD model using RMSE on test sample.

    Args:
        predicted_ratings: full predicted ratings matrix
        user_means: array of user means
        test: test dataframe
        user_map: user_id to index mapping
        item_map: movie_id to index mapping
    Returns:
        RMSE score
    """
    from sklearn.metrics import root_mean_squared_error

    actuals = []
    predictions = []

    test_sample = test.sample(1000, random_state=42)

    for _, row in test_sample.iterrows():
        user_idx = user_map.get(row["user_id"])
        item_idx = item_map.get(row["movie_id"])

        if user_idx is None or item_idx is None:
            continue

        pred = predicted_ratings[user_idx, item_idx] + user_means[user_idx]

        actuals.append(row["rating"])
        predictions.append(pred)

    rmse = root_mean_squared_error(actuals, predictions)
    logger.info(f"SVD RMSE: {rmse:.4f}")
    return rmse


def train_pipeline(
    user_item_matrix: csr_matrix,
    normalized_matrix: csr_matrix,
    user_means: np.ndarray,
    test: pd.DataFrame,
    user_map: dict,
    item_map: dict,
    k_values: list = [10, 20, 50],
    n_factors_list: list = [50, 100, 200]
) -> tuple:
    """
    Train ItemKNN and SVD models, track with MLflow, save best model.

    Args:
        user_item_matrix: original sparse matrix
        normalized_matrix: normalized sparse matrix
        user_means: array of user means
        test: test dataframe
        user_map: user_id to index mapping
        item_map: movie_id to index mapping
        k_values: list of K values for ItemKNN
        n_factors_list: list of factor values for SVD
    Returns:
        tuple: best_model, best_model_type, best_predicted_ratings
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("movie-recommendation-sys")

    best_model = None
    best_model_type = None
    best_rmse = float("inf")
    best_predicted_ratings = None

    # train ItemKNN models
    for k in k_values:
        with mlflow.start_run(run_name=f"ItemKNN_K{k}"):
            model = train_model(normalized_matrix, k=k)
            rmse = _evaluate_itemknn(
                model, user_item_matrix, normalized_matrix,
                user_means, test, user_map, item_map
            )
            mlflow.log_param("model", "ItemKNN")
            mlflow.log_param("k", k)
            mlflow.log_param("metric", "cosine")
            mlflow.log_param("algorithm", "brute")
            mlflow.log_metric("rmse", rmse)
            logger.info(f"ItemKNN K={k} -> RMSE: {rmse:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_model_type = f"itemknn_k{k}"

    # train SVD models
    for n_factors in n_factors_list:
        with mlflow.start_run(run_name=f"SVD_{n_factors}factors"):
            U, sigma, Vt, predicted_ratings = train_svd_model(
                normalized_matrix, n_factors=n_factors
            )
            rmse = _evaluate_svd(
                predicted_ratings, user_means, test, user_map, item_map
            )
            mlflow.log_param("model", "SVD")
            mlflow.log_param("n_factors", n_factors)
            mlflow.log_metric("rmse", rmse)
            logger.info(f"SVD {n_factors} factors -> RMSE: {rmse:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = predicted_ratings
                best_model_type = f"svd_{n_factors}factors"
                best_predicted_ratings = predicted_ratings

    # save best model
    save_model(best_model, model_name=best_model_type)
    logger.info(f"Best model: {best_model_type} RMSE={best_rmse:.4f}")

    return best_model, best_model_type, best_predicted_ratings