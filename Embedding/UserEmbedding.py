# pytorch_user_embeddings.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DataProcessing import DataProcessing
from VectorDataBase.DBHandler import MilvusHandler

def create_user_text(user_row):
    return f"{user_row['Gender']}|{user_row['Age']}|{user_row['Occupation']}|{user_row['Zipcode']}"

class UserEmbeddingProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=768):
        super(UserEmbeddingProjector, self).__init__()
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
    
def embedExistingUser():
    user_pd, _ = DataProcessing.processUserData().toPandas()
    embedding_features = featureExtract(user_pd)
    trainUserData(embedding_features, user_pd)

def featureExtract(users_pd):
    
    # 1. One-hot encoding for Gender
    gender_one_hot = pd.get_dummies(users_pd['Gender'], prefix='Gender')
    print(f"Gender features shape: {gender_one_hot.shape}")
    
    # 2. Bracket encoding for Age
    # Define age brackets
    age_brackets = [0, 18, 25, 35, 45, 55, 65, 100]
    bracket_labels = ['0-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    users_pd['AgeBracket'] = pd.cut(users_pd['Age'], bins=age_brackets, labels=bracket_labels, right=False)
    age_bracket_one_hot = pd.get_dummies(users_pd['AgeBracket'], prefix='Age')
    print(f"Age bracket features shape: {age_bracket_one_hot.shape}")
    
        # 3. Occupation embedding (since occupation is already a code)
    # Get the number of unique occupations
    num_occupations = users_pd['Occupation'].max() + 1 
    
    # Create a simple embedding for occupations
    occupation_embedding_dim = 11
    
    # Create a random embedding matrix for occupations
    np.random.seed(42)  # For reproducibility
    occupation_embedding_matrix = np.random.normal(0, 1, (num_occupations, occupation_embedding_dim))
    
    # Get embeddings for each user's occupation
    occupation_embeddings = np.array([
        occupation_embedding_matrix[int(code)] for code in users_pd['Occupation']
    ])
    print(f"Occupation embeddings shape: {occupation_embeddings.shape}")
    
    users_pd['ZipcodeNum'] = pd.to_numeric(users_pd['Zipcode'], errors='coerce').fillna(0).astype(int)
    
    # Simple geographic hashing (assuming US zipcodes)
    def zipcode_features(zipcode):
        if zipcode == 0:
            return [0, 0]  # Default for missing values
        
        # Extract first 3 digits (geographic area)
        area_code = zipcode // 100
        
        # Create two features: normalized area code and east/west indicator
        normalized_area = area_code / 1000  # Scale to 0-1 range
        east_west = 1 if area_code < 500 else 0  # East/West indicator
        
        return [normalized_area, east_west]
    
    # Apply zipcode encoding
    zipcode_features_list = users_pd['ZipcodeNum'].apply(zipcode_features).tolist()
    zipcode_features_array = np.array(zipcode_features_list)
    print(f"Zipcode features shape: {zipcode_features_array.shape}")
    
    # Combine all features
    gender_features = gender_one_hot.values
    age_features = age_bracket_one_hot.values
    
    # Concatenate all features
    combined_features = np.hstack([
        gender_features,
        age_features,
        occupation_embeddings,
        zipcode_features_array
    ])
    print(f"Combined features shape: {combined_features.shape}")
    
    # Normalize the features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(combined_features)

    return normalized_features

def trainUserData(embedding_features, users_pd):
    
    # Convert to tensor
    initial_features = torch.tensor(embedding_features, dtype=torch.float32)
    
    # Create a mapping from UserID to index
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(users_pd['UserID'])}
    
    # Load movie embeddings (assuming they're already created with BERT)
    # If not, you'll need to create them first
    try:
        movie_db = MilvusHandler(host="localhost", port=19530, collection_name="movies", dim=768)
        movie_db.connect()
        movie_data = movie_db.get_all_entities()
        
        # Create a mapping from MovieID to embedding
        movie_id_to_embedding = {}
        for movie in movie_data:
            movie_id_to_embedding[movie['MovieID']] = torch.tensor(movie['embedding'], dtype=torch.float32)
        
        print(f"Loaded {len(movie_id_to_embedding)} movie embeddings")
    except Exception as e:
        print(f"Error loading movie embeddings: {e}")
        print("Please make sure movie embeddings are created first")
        return None, None
    
    # Initialize the model
    input_dim = initial_features.shape[1]
    model = UserEmbeddingProjector(input_dim=input_dim)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Prepare training data
    train_data = []
    for _, row in users_pd.iterrows():
        user_id = row['UserID']
        movie_id = row['MovieID']
        rating = row['Rating']
        
        # Skip if user or movie not found
        if user_id not in user_id_to_idx or movie_id not in movie_id_to_embedding:
            continue
        
        user_idx = user_id_to_idx[user_id]
        movie_embedding = movie_id_to_embedding[movie_id]
        
        train_data.append((user_idx, movie_embedding, rating))
    
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
            
            # Get user features for this batch
            user_indices = [item[0] for item in batch]
            user_features = initial_features[user_indices]
            
            # Get movie embeddings and ratings
            movie_embeddings = torch.stack([item[1] for item in batch])
            ratings = torch.tensor([item[2] for item in batch], dtype=torch.float32)
            
            # Forward pass
            user_embeddings = model(user_features)
            user_embeddings = nn.functional.normalize(user_embeddings, p=2, dim=1)
            
            # Compute similarity (dot product)
            similarity = torch.sum(user_embeddings * movie_embeddings, dim=1)
            
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
    
    # Generate final embeddings for all users
    with torch.no_grad():
        user_embeddings = model(initial_features)
        user_embeddings = nn.functional.normalize(user_embeddings, p=2, dim=1)

    return users_pd, user_embeddings
    # user_embeddings: tensor of shape [num_users, 32]