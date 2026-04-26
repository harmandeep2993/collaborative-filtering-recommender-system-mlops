"""Main script for Movie Recommendation System using MovieLens 1M dataset."""

from src.data import load_dataset, preprocess_pipeline
from src.features import build_features_pipeline
from src.models import train_pipeline, evaluate_pipeline, predict_pipeline

# step 1 - load datasets
datasets = load_dataset()

# step 2 - preprocess
ratings_train, ratings_test, movies, users = preprocess_pipeline(
    datasets["ratings"],
    datasets["movies"],
    datasets["users"]
)

# step 3 - build features
user_item_matrix, user_map, item_map, user_means, normalized_matrix = build_features_pipeline(ratings_train)
print("Data loading, preprocessing and feature engineering completed.")

# step 4 - train model
best_model, best_model_type, best_predicted_ratings = train_pipeline(
    user_item_matrix,
    normalized_matrix,
    user_means,
    test=ratings_test,
    user_map=user_map,
    item_map=item_map,
    k_values=[10, 20, 50],
    n_factors_list=[50, 100, 200]
)
print(f"Best model: {best_model_type}")
print("Model training completed.")

# step 5 - evaluate best model
report = evaluate_pipeline(
    best_predicted_ratings,
    user_item_matrix,
    normalized_matrix,
    user_means,
    ratings_test,
    user_map,
    item_map,
    best_model_type=best_model_type,
    k=10
)
print("Evaluation completed.")

# step 6 - generate recommendations for user 1
recommendations = predict_pipeline(user_id=1, movies=movies, n=10)
print(f"Top 10 recommendations for User 1:\n{recommendations}")