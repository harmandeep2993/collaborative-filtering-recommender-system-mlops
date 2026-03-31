from src.data import load_dataset
from src.data import preprocess_pipeline
from src.features import build_features_pipeline
from src.models.train import train_pipeline

# step 1 - load
datasets = load_dataset()

# step 2 - preprocess
ratings_train, ratings_test, movies, users = preprocess_pipeline(
    datasets["ratings"],
    datasets["movies"],
    datasets["users"]
)

# step 3 - build features
user_item_matrix, user_map, item_map = build_features_pipeline(ratings_train)

# Step 4 - train model
best_model = train_pipeline(user_item_matrix, test=ratings_test, user_map=user_map, item_map=item_map, k_values=[10, 20, 50])

print("Training completed successfully!")