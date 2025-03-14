# pytorch_bert_embeddings.py
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from tqdm import tqdm  # Add progress bar

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DataProcessing import DataProcessing
from VectorDataBase.DBHandler import MilvusHandler 

class MovieEmbeddingProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=768):
        super(MovieEmbeddingProjector, self).__init__()
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

def embedProcessedData():
    # Load movie data
    movies_pd = DataProcessing.processMovieData().toPandas()
    print(f"Loaded {len(movies_pd)} movies")
    
    # Load ratings data to train the model
    users_pd, ratings_pd = DataProcessing.processUserData().toPandas()
    print(f"Loaded {len(ratings_pd)} ratings")
    
    # Load user data and embeddings
    try:
        user_db = MilvusHandler(host="localhost", port=19530, collection_name="users", dim=768)
        user_db.connect()
        user_data = user_db.get_all_entities()
        
        # Create a mapping from UserID to embedding
        user_id_to_embedding = {}
        for user in user_data:
            user_id_to_embedding[user['UserID']] = torch.tensor(user['embedding'], dtype=torch.float32)
        
        print(f"Loaded {len(user_id_to_embedding)} user embeddings")
    except Exception as e:
        print(f"Error loading user embeddings: {e}")
        print("Please make sure user embeddings are created first")
        return None, None
    
    # Initialize BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()  # Set to evaluation mode
    
    # Create a text field by concatenating title and genres
    movies_pd['Text'] = movies_pd['Title'] + " [SEP] " + movies_pd['Genres'].str.replace('|', ' ')
    
    # Generate BERT embeddings in batches
    batch_size = 8  # Smaller batch size for Mac
    bert_embeddings = []
    
    # Use tqdm for progress tracking
    with torch.no_grad():  # Disable gradient calculation for inference
        for i in tqdm(range(0, len(movies_pd), batch_size), desc="Generating BERT embeddings"):
            batch_texts = movies_pd['Text'][i:i+batch_size].tolist()
            
            # Tokenize the entire batch at once
            inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, 
                              max_length=512, padding='max_length')
            
            # Get embeddings for the batch
            outputs = bert_model(**inputs)
            
            # Use the [CLS] token embedding as the sentence representation
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Add to our list of embeddings
            bert_embeddings.append(cls_embeddings.detach().numpy())
    
    # Concatenate all batches
    bert_embeddings = np.vstack(bert_embeddings)
    
    # Convert to tensor
    bert_embeddings_tensor = torch.tensor(bert_embeddings, dtype=torch.float32)
    
    # Create a mapping from MovieID to index
    movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies_pd['MovieID'])}
    
    # Initialize the model
    input_dim = bert_embeddings_tensor.shape[1]  # BERT dimension (768)
    model = MovieEmbeddingProjector(input_dim=input_dim)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Prepare training data
    train_data = []
    for _, row in ratings_pd.iterrows():
        user_id = row['UserID']
        movie_id = row['MovieID']
        rating = row['Rating']
        
        # Skip if user or movie not found
        if user_id not in user_id_to_embedding or movie_id not in movie_id_to_idx:
            continue
        
        user_embedding = user_id_to_embedding[user_id]
        movie_idx = movie_id_to_idx[movie_id]
        
        train_data.append((movie_idx, user_embedding, rating))
    
    print(f"Prepared {len(train_data)} training examples")
    
    # Training loop
    num_epochs = 10
    batch_size = 64
    
    for epoch in range(num_epochs):
        # Shuffle training data
        np.random.shuffle(train_data)
        
        # Process in batches
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            # Get movie features for this batch
            movie_indices = [item[0] for item in batch]
            movie_features = bert_embeddings_tensor[movie_indices]
            
            # Get user embeddings and ratings
            user_embeddings = torch.stack([item[1] for item in batch])
            ratings = torch.tensor([item[2] for item in batch], dtype=torch.float32)
            
            # Forward pass
            movie_embeddings = model(movie_features)
            movie_embeddings = nn.functional.normalize(movie_embeddings, p=2, dim=1)
            
            # Compute similarity (dot product)
            similarity = torch.sum(movie_embeddings * user_embeddings, dim=1)
            
            # Scale similarity to rating range (1-5)
            predicted_ratings = 1 + 4 * (similarity + 1) / 2  # Map from [-1,1] to [1,5]
            
            # Compute loss
            loss = criterion(predicted_ratings, ratings)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Print epoch statistics
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Generate final embeddings for all movies
    with torch.no_grad():
        movie_embeddings = model(bert_embeddings_tensor)
        movie_embeddings = nn.functional.normalize(movie_embeddings, p=2, dim=1)
    
    # Get the final dimension
    embedding_dim = movie_embeddings.shape[1]
    print(f"Final movie embeddings shape: {movie_embeddings.shape}")
    
    # Store in vector database
    print("#### insert movie data ####")
    db_handler = MilvusHandler(host="localhost", port=19530, collection_name="movies", dim=embedding_dim)
    db_handler.insert_data(movies_pd, movie_embeddings)

    return movies_pd, movie_embeddings

    # final_embeddings is a [num_movies x 64] tensor ready to be stored in Milvus
