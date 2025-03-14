import torch
import pandas as pd
import sys
import os
import numpy as np
from pymilvus import utility

# Add the project root to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the Embedding package
from Embedding import UserEmbedding
from Embedding import MovieEmbedding
from VectorDataBase.DBHandler import MilvusHandler

def recommend_movies_for_user(user_id, top_k=5):
    # Initialize Milvus handlers
    user_db_handler = MilvusHandler(host="127.0.0.1", port=19530, collection_name="users_final", dim=768)
    movie_db_handler = MilvusHandler(host="127.0.0.1", port=19530, collection_name="movies_final", dim=768)
    
    # Retrieve user data from Milvus
    users_data = user_db_handler.get_entities(expr=f"UserID == {user_id}")
    if not users_data or len(users_data) == 0:
        print(f"User {user_id} not found in Milvus")
        return []
    
    user_embed = users_data[0]['embedding']
    
    # Use Milvus to search for similar movies
    search_results = movie_db_handler.search(user_embed, top_k)
    # Get movie details for the results
    movie_ids = [hit.id for hit in search_results[0]]
    movies_data = movie_db_handler.get_entities(
        expr=f"MovieID in {movie_ids}", 
        output_fields=["Title"]
    )
    
    recommended_titles = [movie['Title'] for movie in movies_data]
    return recommended_titles

# Example usage:
if __name__ == "__main__":
    
    user_id_example = 13  # Replace with a valid UserID from your dataset
    recommendations = recommend_movies_for_user(user_id_example)
    print("Recommended Movies for User", user_id_example, ":", recommendations)
