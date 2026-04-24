# tests/test_data.py

"""
Unit tests for data loading and preprocessing functions.
"""

import pytest
import pandas as pd
from src.data import load_dataset, preprocess_pipeline

# Test that the load_dataset function returns a dictionary with the expected keys
def test_load_dataset_retturns_correct_keys():

    datasets= load_dataset()
    assert "movies" in datasets
    assert "ratings" in datasets
    assert "users" in datasets

def test_load_dataset_correct_shapes():
    datasets = load_dataset()
    assert datasets["ratings"].shape[0] == 1000209
    assert datasets["movies"].shape[0] == 3883
    assert datasets["users"].shape[0] == 6040

def test_load_dataset_correct_columns():
    datasets = load_dataset()
    assert "user_id" in datasets["ratings"].columns
    assert "movie_id" in datasets["ratings"].columns
    assert "rating" in datasets["ratings"].columns

def test_preprocess_pipeline_returns_correct_shapes():
    datasets = load_dataset()
    train, test, movies, users = preprocess_pipeline(
        datasets["ratings"],
        datasets["movies"],
        datasets["users"]
    )
    assert train.shape[0] > test.shape[0]
    assert test.shape[0] > 0

def test_preprocess_pipeline_no_missing_values():
    datasets = load_dataset()
    train, test, movies, users = preprocess_pipeline(
        datasets["ratings"],
        datasets["movies"],
        datasets["users"]
    )
    assert train.isnull().sum().sum() == 0
    assert test.isnull().sum().sum() == 0
