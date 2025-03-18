import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DataProcessing import DataProcessing
from VectorDataBase.DBHandler import MilvusHandler

class EmbeddingProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=768):
        super(EmbeddingProjector, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def extract_movie_features(movies_pd, tokenizer_info):
    """Extract BERT features from movie data using saved tokenizer information"""
    print("Initializing BERT model...")
    # Initialize BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_info['tokenizer_name'])
    model = BertModel.from_pretrained(tokenizer_info['tokenizer_name'])
    
    # Combine title and genres for each movie
    movies_pd['text'] = movies_pd['Title'] + " " + movies_pd['Genres'].str.replace('|', ' ')
    
    # Generate BERT embeddings
    bert_embeddings = []
    batch_size = 32  # Process in small batches to avoid memory issues
    
    print(f"Generating BERT embeddings for {len(movies_pd)} movies...")
    for i in tqdm(range(0, len(movies_pd), batch_size), desc="Extracting features"):
        batch_texts = movies_pd['text'].iloc[i:i+batch_size].tolist()
        
        # Tokenize and prepare input
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=tokenizer_info['max_length']
        )
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the [CLS] token embedding as the sentence representation
            batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        
        bert_embeddings.append(batch_embeddings)
    
    # Combine all batches
    bert_embeddings = np.vstack(bert_embeddings)
    print(f"BERT embeddings shape: {bert_embeddings.shape}")
    
    return bert_embeddings

def generate_movie_embeddings(model_path, tokenizer_info_path, output_collection, recreate_collection=False):
    """Generate embeddings for all movies using the trained model"""
    print("Loading movie data...")
    # Load movie data
    movies_pd = DataProcessing.processMovieData().toPandas()
    print(f"Loaded {len(movies_pd)} movies")
    
    with open(tokenizer_info_path, "rb") as f:
        tokenizer_info = pickle.load(f)
    
    # Extract BERT features
    bert_embeddings = extract_movie_features(movies_pd, tokenizer_info)
    
    input_dim = bert_embeddings.shape[1]
    model = EmbeddingProjector(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Generate embeddings
    print("Generating final embeddings...")
    bert_embeddings_tensor = torch.tensor(bert_embeddings, dtype=torch.float32)
    
    with torch.no_grad():
        movie_embeddings = model(bert_embeddings_tensor)
        movie_embeddings = nn.functional.normalize(movie_embeddings, p=2, dim=1)
    
    movie_embeddings_np = movie_embeddings.numpy()
    print(f"Final movie embeddings shape: {movie_embeddings_np.shape}")
    
    # Connect to Milvus
    print("Connecting to Milvus...")
    db_handler = MilvusHandler(host="localhost", port=19530, collection_name=output_collection, dim=768, dataType="movie", needToReset=recreate_collection)
    
    db_handler.load_collection()
    
    db_handler.insert_movie_data(movies_pd, movie_embeddings)
    
    connections.disconnect("default")
    
    print("Movie embedding generation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate movie embeddings using trained model")
    parser.add_argument("--model", type=str, default="models/movie_model_final.pt", help="Path to the trained movie model")
    parser.add_argument("--tokenizer_info", type=str, default="models/movie_tokenizer_info.pkl", help="Path to the tokenizer info file")
    parser.add_argument("--output_collection", type=str, default="movies_final_for_cold_start", help="Name of the output Milvus collection")
    parser.add_argument("--recreate", action="store_true", help="Recreate the collection if it exists")
    
    args = parser.parse_args()
    
    generate_movie_embeddings(
        model_path="models/movie_model_final.pt",
        tokenizer_info_path="models/movie_tokenizer_info.pkl",
        output_collection="movies_final_for_cold_start",
        recreate_collection=True
    )
