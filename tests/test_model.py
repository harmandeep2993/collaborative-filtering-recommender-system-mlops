#tests/test_model.py

"""
Unit tests for model training and prediction.
"""

import pytest
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz
from src.models.predict import load_model, load_artifacts, recommend_movies


def test_model_loads_correctly():
    model = load_model()
    assert model is not None


def test_model_has_correct_neighbors():
    model = load_model()
    assert model.n_neighbors == 50


def test_artifacts_load_correctly():
    user_item_matrix, user_map, item_map, idx_to_item = load_artifacts()
    assert user_item_matrix is not None
    assert len(user_map) > 0
    assert len(item_map) > 0
    assert len(idx_to_item) > 0


def test_matrix_correct_shape():
    user_item_matrix, user_map, item_map, idx_to_item = load_artifacts()
    assert user_item_matrix.shape[0] == 6040
    assert user_item_matrix.shape[1] == 3260


def test_recommendations_returns_correct_number():
    from src.data import load_dataset
    datasets = load_dataset()
    model = load_model()
    user_item_matrix, user_map, item_map, idx_to_item = load_artifacts()

    recommendations = recommend_movies(
        user_id=1,
        model=model,
        user_item_matrix=user_item_matrix,
        user_map=user_map,
        item_map=item_map,
        idx_to_item=idx_to_item,
        movies=datasets["movies"],
        n=10
    )

    assert recommendations is not None
    assert len(recommendations) == 10


def test_invalid_user_returns_none():
    from src.data import load_dataset
    datasets = load_dataset()
    model = load_model()
    user_item_matrix, user_map, item_map, idx_to_item = load_artifacts()

    recommendations = recommend_movies(
        user_id=99999,
        model=model,
        user_item_matrix=user_item_matrix,
        user_map=user_map,
        item_map=item_map,
        idx_to_item=idx_to_item,
        movies=datasets["movies"],
        n=10
    )

    assert recommendations is None