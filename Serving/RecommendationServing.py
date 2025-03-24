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
from Serving.ColdStartUserEmbeddingGenerator import generate_newuser_embeddings, User

def recommend_movies_for_user(user_id, top_k=5):
    # Initialize Milvus handlers
    user_db_handler = MilvusHandler(host="127.0.0.1", port=19530, collection_name="users_final", dim=768)
    
    # Retrieve user data from Milvus
    users_data = user_db_handler.get_entities(expr=f"UserID == {user_id}")
    if not users_data or len(users_data) == 0:
        print(f"need To use initial user model")
        return []
    
    user_embed = users_data[0]['embedding']
    print(f"userID: {user_id}")
    print(f"user_embed: {user_embed}")
    return search_movie_with_embedding(user_embed, top_k)
    

def recommend_movies_for_user_cold_start(new_user_embedding, top_k=5):
    return search_movie_with_embedding(new_user_embedding, top_k)
    

def search_movie_with_embedding(user_embedding, top_k=5):
    # Use Milvus to search for similar movies
    movie_db_handler = MilvusHandler(host="127.0.0.1", port=19530, collection_name="movies_final", dim=768)

    search_results = movie_db_handler.search(user_embedding, top_k)
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
    # existing user
    # user_id_example = 543
    # recommendations = recommend_movies_for_user(user_id_example)

    # new user 
    user = User(user_id=1000100, user_gender="F", user_age=7, user_occupation=12, user_zipcode=43920)
    user_embedding_vector = generate_newuser_embeddings("models/user_model_final.pt", user)
    # print(f"embedding vector {user_embedding_vector}")
    recommendations = recommend_movies_for_user_cold_start(user_embedding_vector)


    print(recommendations)
    
