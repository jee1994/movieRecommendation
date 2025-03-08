import torch
import pandas as pd
import sys
import os
import numpy as np

# Add the project root to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the Embedding package
from Embedding import UserEmbedding
from Embedding import MovieEmbedding

def recommend_movies_for_user(user_id, top_k=5):
    # Load embeddings and data
    users_pd, user_embeddings = UserEmbedding.embedUserData()
    movies_pd, final_movie_embeddings = MovieEmbedding.embedProcessedData()
    
    # Find the user embedding based on user_id
    user_index = users_pd[users_pd["UserID"] == user_id].index[0]
    user_embed = user_embeddings[user_index]
    
    # For demonstration, compute cosine similarity (dot product as embeddings are normalized)
    similarities = torch.matmul(final_movie_embeddings, user_embed)
    
    # Get indices of top-k movies
    top_indices = torch.topk(similarities, top_k).indices
    recommended_titles = movies_pd.iloc[top_indices.tolist()]["Title"].tolist()
    
    return recommended_titles

# Example usage:
if __name__ == "__main__":
    user_id_example = 1  # Replace with a valid UserID from your dataset
    recommendations = recommend_movies_for_user(user_id_example)
    print("Recommended Movies for User", user_id_example, ":", recommendations)
