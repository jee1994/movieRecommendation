import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os
import shutil
import argparse
import pickle
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DataProcessing import DataProcessing
from VectorDataBase.DBHandler import MilvusHandler

class SimpleProjector(nn.Module):
    def __init__(self, input_dim, output_dim=768):
        super(SimpleProjector, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.fc(x)

class EmbeddingProjector(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], output_dim=768):
        super(EmbeddingProjector, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class EnhancedMovieEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=768, dropout_rate=0.3):
        super(EnhancedMovieEmbedding, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # Optional: Add genre-specific embeddings
        self.use_genre_embeddings = False
        if self.use_genre_embeddings:
            self.genre_embeddings = nn.Embedding(20, 64)  # Assuming 20 genres
            self.genre_projection = nn.Linear(64, output_dim)
    
    def forward(self, x, genres=None):
        # Main content embedding
        content_embedding = self.model(x)
        
        # Add genre information if available
        if self.use_genre_embeddings and genres is not None:
            genre_embedding = self.genre_embeddings(genres)
            genre_embedding = self.genre_projection(genre_embedding)
            content_embedding = content_embedding + genre_embedding
        
        # Normalize output
        output = nn.functional.normalize(content_embedding, p=2, dim=1)
        return output

def create_initial_user_embeddings():
    """Create initial user embeddings based only on demographic features"""
    print("\n=== Creating Initial User Embeddings ===")
    
    # Load user data
    users_pd, _ = DataProcessing.processUserData()
    users_pd = users_pd.toPandas()
    print(f"Loaded {len(users_pd)} users")
    
    # Feature extraction
    features, feature_info = extract_user_features(users_pd)
    
    # Save feature extraction info for later use
    os.makedirs("models", exist_ok=True)
    with open("models/user_feature_info.pkl", "wb") as f:
        pickle.dump(feature_info, f)
    
    # Create a simple projection to 768 dimensions
    input_dim = features.shape[1]
    model = SimpleProjector(input_dim)
    
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float32)
        user_embeddings = model(features_tensor)
        user_embeddings = nn.functional.normalize(user_embeddings, p=2, dim=1)
    
    # Save the model
    torch.save(model.state_dict(), "models/initial_user_model.pt")
    print(f"Saved initial user model to models/initial_user_model.pt")
    
    # Store in vector database
    db_handler = MilvusHandler(host="localhost", port=19530, collection_name="model_initial_users", dim=768, dataType="user", needToReset=True)
    db_handler.insert_user_data(users_pd, user_embeddings)
    
    return users_pd, user_embeddings

def extract_user_features(users_pd):
    """Extract features from user data and return feature information for reuse"""
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
    
    # 3. Occupation embedding
    num_occupations = users_pd['Occupation'].max() + 1
    occupation_embedding_dim = 11
    
    # Create a random embedding matrix for occupations
    np.random.seed(42)  # For reproducibility
    occupation_embedding_matrix = np.random.normal(0, 1, (num_occupations, occupation_embedding_dim))
    
    # Get embeddings for each user's occupation
    occupation_embeddings = np.array([
        occupation_embedding_matrix[int(code)] for code in users_pd['Occupation']
    ])
    print(f"Occupation embeddings shape: {occupation_embeddings.shape}")
    
    # 4. Zipcode features
    users_pd['ZipcodeNum'] = pd.to_numeric(users_pd['Zipcode'], errors='coerce').fillna(0).astype(int)
    
    def zipcode_features(zipcode):
        if zipcode == 0:
            return [0, 0]
        area_code = zipcode // 100
        normalized_area = area_code / 1000
        east_west = 1 if area_code < 500 else 0
        return [normalized_area, east_west]
    
    zipcode_features_list = users_pd['ZipcodeNum'].apply(zipcode_features).tolist()
    zipcode_features_array = np.array(zipcode_features_list)
    print(f"Zipcode features shape: {zipcode_features_array.shape}")
    
    # Combine all features
    combined_features = np.hstack([
        gender_one_hot.values,
        age_bracket_one_hot.values,
        occupation_embeddings,
        zipcode_features_array
    ])
    print(f"Combined features shape: {combined_features.shape}")
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(combined_features)
    
    # Store feature information for later use
    feature_info = {
        'gender_columns': list(gender_one_hot.columns),
        'age_brackets': age_brackets,
        'age_bracket_labels': bracket_labels,
        'age_columns': list(age_bracket_one_hot.columns),
        'num_occupations': num_occupations,
        'occupation_embedding_dim': occupation_embedding_dim,
        'occupation_embedding_matrix': occupation_embedding_matrix,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_
    }
    
    return normalized_features, feature_info

def create_initial_movie_embeddings():
    """Create initial movie embeddings based on BERT embeddings of title and genre"""
    print("\n=== Creating Initial Movie Embeddings ===")
    
    # Load movie data
    movies_pd = DataProcessing.processMovieData().toPandas()
    print(f"Loaded {len(movies_pd)} movies")
    
    # Generate BERT embeddings
    bert_embeddings, tokenizer_info = extract_movie_features(movies_pd)
    
    # Save tokenizer info for later use
    with open("models/movie_tokenizer_info.pkl", "wb") as f:
        pickle.dump(tokenizer_info, f)
    
    # Create a simple projection (identity in this case)
    input_dim = bert_embeddings.shape[1]
    model = SimpleProjector(input_dim)
    
    with torch.no_grad():
        bert_embeddings_tensor = torch.tensor(bert_embeddings, dtype=torch.float32)
        movie_embeddings = model(bert_embeddings_tensor)
        movie_embeddings = nn.functional.normalize(movie_embeddings, p=2, dim=1)
    
    # Save the model
    torch.save(model.state_dict(), "models/initial_movie_model.pt")
    print(f"Saved initial movie model to models/initial_movie_model.pt")
    
    # Store in vector database
    db_handler = MilvusHandler(host="localhost", port=19530, collection_name="model_initial_movies", dim=768, dataType="movie", needToReset=True)
    db_handler.insert_movie_data(movies_pd, movie_embeddings)
    
    return movies_pd, movie_embeddings

def extract_movie_features(movies_pd):
    """Extract BERT features from movie data"""
    # Initialize BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Combine title and genres for each movie
    movies_pd['text'] = movies_pd['Title'] + " " + movies_pd['Genres'].str.replace('|', ' ')
    
    # Generate BERT embeddings
    bert_embeddings = []
    batch_size = 32  # Process in small batches to avoid memory issues
    
    for i in tqdm(range(0, len(movies_pd), batch_size), desc="Generating BERT embeddings"):
        batch_texts = movies_pd['text'].iloc[i:i+batch_size].tolist()
        
        # Tokenize and prepare input
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the [CLS] token embedding as the sentence representation
            batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        
        bert_embeddings.append(batch_embeddings)
    
    # Combine all batches
    bert_embeddings = np.vstack(bert_embeddings)
    print(f"BERT embeddings shape: {bert_embeddings.shape}")
    
    # Store tokenizer info
    tokenizer_info = {
        'tokenizer_name': 'bert-base-uncased',
        'max_length': 128
    }
    
    return bert_embeddings, tokenizer_info

def train_user_embeddings(iteration, movie_collection):
    """Train user embeddings with contrastive learning"""
    print(f"\n=== Training User Embeddings (Iteration {iteration}) ===")
    print(f"Using movie collection: {movie_collection}")
    
    # Load user data
    users_pd, ratings_pd = DataProcessing.processUserData()
    users_pd = users_pd.toPandas()
    ratings_pd = ratings_pd.toPandas()
    print(f"Loaded {len(users_pd)} users")
    
    # Load movie embeddings
    try:
        movie_db = MilvusHandler(host="localhost", port=19530, collection_name=movie_collection, dim=768, dataType="movie")
        movie_data = movie_db.get_all_movie_entities()
        
        # Create a mapping from MovieID to embedding
        movie_id_to_embedding = {}
        for movie in movie_data:
            movie_id_to_embedding[movie['MovieID']] = torch.tensor(movie['embedding'], dtype=torch.float32)
        
        print(f"Loaded {len(movie_id_to_embedding)} movie embeddings")
    except Exception as e:
        print(f"Error loading movie embeddings: {e}")
        return None, None
    
    # Extract user features
    with open("models/user_feature_info.pkl", "rb") as f:
        feature_info = pickle.load(f)
    
    # Re-extract features using the same process
    features = extract_user_features_with_info(users_pd, feature_info)
    
    # Prepare training data
    train_data = []
    for _, row in ratings_pd.iterrows():
        user_id = row['UserID']
        movie_id = row['MovieID']
        rating = row['Rating']
        
        # Skip if movie embedding is not available
        if movie_id not in movie_id_to_embedding:
            continue
        
        # Get user index (assuming UserID is 1-indexed)
        user_idx = user_id - 1
        
        # Skip if user index is out of bounds
        if user_idx < 0 or user_idx >= len(features):
            continue
        
        # Add to training data
        train_data.append((user_idx, movie_id_to_embedding[movie_id], rating))
    
    print(f"Prepared {len(train_data)} training examples")
    
    # Initialize the model
    input_dim = features.shape[1]
    model = EmbeddingProjector(input_dim=input_dim)
    
    # Load previous model if available
    if iteration > 0:
        prev_model_path = f"models/user_model_iter{iteration-1}.pt"
        if os.path.exists(prev_model_path):
            print(f"Loading previous model from {prev_model_path}")
            model.load_state_dict(torch.load(prev_model_path))
    
    # Convert features to tensor
    features_tensor = torch.tensor(features, dtype=torch.float32)
    
    # Training parameters
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    num_epochs = 10
    batch_size = 64
    
    # Contrastive learning parameters
    contrastive_weight = 0.2  # Weight for contrastive loss
    temperature = 0.1  # Temperature parameter for similarity scaling
    
    # Training loop
    for epoch in range(num_epochs):
        # Shuffle training data
        indices = np.random.permutation(len(train_data))
        
        # Process in batches
        total_loss = 0
        total_rating_loss = 0
        total_contrastive_loss = 0
        num_batches = 0
        
        model.train()
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            
            # Get user features for this batch
            user_indices = [train_data[idx][0] for idx in batch_indices]
            user_features = features_tensor[user_indices]
            
            # Get movie embeddings and ratings
            movie_embeddings = torch.stack([train_data[idx][1] for idx in batch_indices])
            ratings = torch.tensor([train_data[idx][2] for idx in batch_indices], dtype=torch.float32)
            
            # Forward pass
            user_embeddings = model(user_features)
            user_embeddings = nn.functional.normalize(user_embeddings, p=2, dim=1)
            
            # Compute similarity (dot product)
            similarity = torch.sum(user_embeddings * movie_embeddings, dim=1)
            
            # Scale similarity to rating range (1-5)
            predicted_ratings = 1 + 4 * (similarity + 1) / 2
            
            # Rating prediction loss
            rating_loss = criterion(predicted_ratings, ratings)
            
            # Contrastive loss
            contrastive_loss = compute_contrastive_loss(
                user_embeddings, user_features, temperature=temperature
            )
            
            # Combined loss
            loss = rating_loss + contrastive_weight * contrastive_loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_rating_loss += rating_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            num_batches += 1
        
        # Print epoch statistics
        avg_loss = total_loss / num_batches
        avg_rating_loss = total_rating_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f} "
              f"(Rating: {avg_rating_loss:.4f}, Contrastive: {avg_contrastive_loss:.4f})")
    
    model.eval()
    # Generate final embeddings for all users
    with torch.no_grad():
        user_embeddings = model(features_tensor)
        user_embeddings = nn.functional.normalize(user_embeddings, p=2, dim=1)
    
    # Get the final dimension
    embedding_dim = user_embeddings.shape[1]
    print(f"Final user embeddings shape: {user_embeddings.shape}")
    
    # Save the model
    model_path = f"models/user_model_iter{iteration}.pt" if iteration > 0 else "models/user_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved user model to {model_path}")
    
    # Store in vector database
    collection_name = f"users_iter{iteration}" if iteration > 0 else "users"
    print(f"#### insert user data to {collection_name} ####")
    db_handler = MilvusHandler(host="localhost", port=19530, collection_name=collection_name, dim=embedding_dim, dataType="user")
    db_handler.insert_user_data(users_pd, user_embeddings)
    
    return users_pd, user_embeddings

def extract_user_features_with_info(users_pd, feature_info):
    """Extract user features using saved feature information"""
    # 1. One-hot encoding for Gender
    gender_one_hot = pd.get_dummies(users_pd['Gender'], prefix='Gender')
    
    # Ensure all expected columns are present
    for col in feature_info['gender_columns']:
        if col not in gender_one_hot.columns:
            gender_one_hot[col] = 0
    
    # Reorder columns to match training
    gender_one_hot = gender_one_hot[feature_info['gender_columns']]
    
    # 2. Bracket encoding for Age
    users_pd['AgeBracket'] = pd.cut(
        users_pd['Age'], 
        bins=feature_info['age_brackets'], 
        labels=feature_info['age_bracket_labels'], 
        right=False
    )
    age_bracket_one_hot = pd.get_dummies(users_pd['AgeBracket'], prefix='Age')
    
    # Ensure all expected columns are present
    for col in feature_info['age_columns']:
        if col not in age_bracket_one_hot.columns:
            age_bracket_one_hot[col] = 0
    
    # Reorder columns to match training
    age_bracket_one_hot = age_bracket_one_hot[feature_info['age_columns']]
    
    # 3. Occupation embedding
    occupation_embedding_matrix = feature_info['occupation_embedding_matrix']
    
    # Get embeddings for each user's occupation
    occupation_embeddings = np.array([
        occupation_embedding_matrix[min(int(code), len(occupation_embedding_matrix)-1)] 
        for code in users_pd['Occupation']
    ])
    
    # 4. Zipcode features
    users_pd['ZipcodeNum'] = pd.to_numeric(users_pd['Zipcode'], errors='coerce').fillna(0).astype(int)
    
    def zipcode_features(zipcode):
        if zipcode == 0:
            return [0, 0]
        area_code = zipcode // 100
        normalized_area = area_code / 1000
        east_west = 1 if area_code < 500 else 0
        return [normalized_area, east_west]
    
    zipcode_features_list = users_pd['ZipcodeNum'].apply(zipcode_features).tolist()
    zipcode_features_array = np.array(zipcode_features_list)
    
    # Combine all features
    combined_features = np.hstack([
        gender_one_hot.values,
        age_bracket_one_hot.values,
        occupation_embeddings,
        zipcode_features_array
    ])
    
    # Apply the same normalization as during training
    normalized_features = (combined_features - feature_info['scaler_mean']) / feature_info['scaler_scale']
    
    return normalized_features

def train_movie_embeddings(iteration, user_collection):
    """Train movie embeddings based on user embeddings from the specified collection"""
    print(f"\n=== Training Movie Embeddings (Iteration {iteration}) ===")
    print(f"Using user collection: {user_collection}")
    
    # Load movie data
    movies_pd = DataProcessing.processMovieData().toPandas()
    _, ratings_pd = DataProcessing.processUserData()
    ratings_pd = ratings_pd.toPandas()
    print(f"Loaded {len(movies_pd)} movies")
    
    # Load user embeddings
    try:
        user_db = MilvusHandler(host="localhost", port=19530, collection_name=user_collection, dim=768, dataType="user")
        user_db.connect()
        user_data = user_db.get_all_user_entities()
        
        # Create a mapping from UserID to embedding
        user_id_to_embedding = {}
        for user in user_data:
            user_id_to_embedding[user['UserID']] = torch.tensor(user['embedding'], dtype=torch.float32)
        
        print(f"Loaded {len(user_id_to_embedding)} user embeddings")
    except Exception as e:
        print(f"Error loading user embeddings: {e}")
        return None, None
    
    # Extract movie features (BERT embeddings)
    with open("models/movie_tokenizer_info.pkl", "rb") as f:
        tokenizer_info = pickle.load(f)
    
    # Re-extract BERT features
    bert_embeddings = extract_movie_features_with_info(movies_pd, tokenizer_info)
    
    # Prepare training data
    train_data = []
    for _, row in ratings_pd.iterrows():
        user_id = row['UserID']
        movie_id = row['MovieID']
        rating = row['Rating']
        
        # Skip if user embedding is not available
        if user_id not in user_id_to_embedding:
            continue
        
        # Get movie index (assuming MovieID is 1-indexed)
        movie_idx = movie_id - 1
        
        # Skip if movie index is out of bounds
        if movie_idx < 0 or movie_idx >= len(bert_embeddings):
            continue
        
        # Add to training data
        train_data.append((movie_idx, user_id_to_embedding[user_id], rating))
    
    print(f"Prepared {len(train_data)} training examples")
    
    # Initialize the model
    input_dim = bert_embeddings.shape[1]
    model = EnhancedMovieEmbedding(input_dim=input_dim)

    if iteration > 0:
        prev_model_path = f"models/movie_model_iter{iteration-1}.pt"
        if os.path.exists(prev_model_path):
            print(f"Loading previous model from {prev_model_path}")
            model.load_state_dict(torch.load(prev_model_path))
    
    # Convert BERT embeddings to tensor
    bert_embeddings_tensor = torch.tensor(bert_embeddings, dtype=torch.float32)
    
    # Training parameters
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    batch_size = 64
    
    # Create a pool of negative users for contrastive learning
    all_user_embeddings = torch.stack(list(user_id_to_embedding.values()))
    model.train()
    # Training loop
    for epoch in range(num_epochs):
        # Shuffle training data
        np.random.shuffle(train_data)
        
        # Process in batches
        total_loss = 0
        total_rating_loss = 0
        total_ranking_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            # Get movie features for this batch
            movie_indices = [item[0] for item in batch]
            movie_features = bert_embeddings_tensor[movie_indices]
            
            # Get user embeddings and ratings
            user_embeddings = torch.stack([item[1] for item in batch])
            ratings = torch.tensor([item[2] for item in batch], dtype=torch.float32)
            
            # Sample negative users (who didn't rate these movies)
            negative_user_embeddings = sample_negative_users(user_embeddings, all_user_embeddings, k=5)
            
            # Forward pass
            movie_embeddings = model(movie_features)
            
            # Compute hybrid loss
            loss, rating_loss, ranking_loss, _ = hybrid_movie_embedding_loss(
                movie_embeddings, 
                user_embeddings, 
                ratings,
                negative_user_embeddings,
                alpha=0.4  # Adjust based on your priorities
            )
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_rating_loss += rating_loss.item()
            total_ranking_loss += ranking_loss.item()
            num_batches += 1
        
        # Print epoch statistics
        avg_loss = total_loss / num_batches
        avg_rating_loss = total_rating_loss / num_batches
        avg_ranking_loss = total_ranking_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f} "
              f"(Rating: {avg_rating_loss:.4f}, Ranking: {avg_ranking_loss:.4f})")
    
    model.eval()
    # Generate final embeddings for all movies
    with torch.no_grad():
        movie_embeddings = model(bert_embeddings_tensor)
        movie_embeddings = nn.functional.normalize(movie_embeddings, p=2, dim=1)
    
    # Get the final dimension
    embedding_dim = movie_embeddings.shape[1]
    print(f"Final movie embeddings shape: {movie_embeddings.shape}")
    
    # Save the model
    model_path = f"models/movie_model_iter{iteration}.pt" if iteration > 0 else "models/movie_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved movie model to {model_path}")
    
    # Store in vector database
    collection_name = f"movies_iter{iteration}" if iteration > 0 else "movies"
    print(f"#### insert movie data to {collection_name} ####")
    db_handler = MilvusHandler(host="localhost", port=19530, collection_name=collection_name, dim=embedding_dim, dataType="movie")
    db_handler.insert_movie_data(movies_pd, movie_embeddings)
    
    return movies_pd, movie_embeddings

def extract_movie_features_with_info(movies_pd, tokenizer_info):
    """Extract BERT features from movie data using saved tokenizer information"""
    # Initialize BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_info['tokenizer_name'])
    model = BertModel.from_pretrained(tokenizer_info['tokenizer_name'])
    
    # Combine title and genres for each movie
    movies_pd['text'] = movies_pd['Title'] + " " + movies_pd['Genres'].str.replace('|', ' ')
    
    # Generate BERT embeddings
    bert_embeddings = []
    batch_size = 32  # Process in small batches to avoid memory issues
    
    for i in tqdm(range(0, len(movies_pd), batch_size), desc="Generating BERT embeddings"):
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

def copy_collection(source, target):
    """Copy data from source collection to target collection"""
    data_type = "user" if "user" in source else "movie"
    try:
        # Connect to source collection
        source_db = MilvusHandler(host="localhost", port=19530, collection_name=source, dim=768, dataType=data_type)
        source_db.connect()
        
        # Get all entities from source
        entities = source_db.get_all_user_entities() if data_type == "user" else source_db.get_all_movie_entities()
        
        if not entities:
            print(f"No entities found in {source}")
            return False
        
        
        # Extract data and embeddings
        data = {}
        if data_type == "user":
            data['UserID'] = [entity['UserID'] for entity in entities]
        else:
            data['MovieID'] = [entity['MovieID'] for entity in entities]
        
        data_df = pd.DataFrame(data)
        embeddings = torch.tensor(np.array([entity['embedding'] for entity in entities]))
        
        # Create target collection and insert data
        target_db = MilvusHandler(host="localhost", port=19530, collection_name=target, dim=768, dataType=data_type)
        if data_type == "user": 
            target_db.insert_user_data(data_df, embeddings)
        else:
            target_db.insert_movie_data(data_df, embeddings)
        
        print(f"Copied {len(entities)} entities from {source} to {target}")
        return True
    except Exception as e:
        print(f"Error copying collection: {e}")
        return False

def copy_final_models(iteration):
    """Copy the final models to the final model files"""
    try:
        # Copy user model
        user_model_src = f"models/user_model_iter{iteration}.pt" if iteration > 0 else "models/user_model.pt"
        shutil.copy(user_model_src, "models/user_model_final.pt")
        
        # Copy movie model
        movie_model_src = f"models/movie_model_iter{iteration}.pt" if iteration > 0 else "models/movie_model.pt"
        shutil.copy(movie_model_src, "models/movie_model_final.pt")
        
        print(f"Copied final models from iteration {iteration}")
        return True
    except Exception as e:
        print(f"Error copying final models: {e}")
        return False

def compute_contrastive_loss(embeddings, features, temperature=0.1):
    """
    Compute contrastive loss to ensure different users have different embeddings
    
    Args:
        embeddings: Normalized user embeddings (batch_size x embedding_dim)
        features: User demographic features (batch_size x feature_dim)
        temperature: Temperature parameter for similarity scaling
    
    Returns:
        Contrastive loss value
    """
    batch_size = embeddings.shape[0]
    
    # Compute pairwise embedding similarities
    # sim(i,j) = e_i^T e_j / temperature
    similarities = torch.mm(embeddings, embeddings.t()) / temperature
    
    # Compute feature distances (demographic differences)
    # We want users with different demographics to have different embeddings
    feature_distances = torch.cdist(features, features, p=2)
    
    # Normalize feature distances to [0, 1]
    if feature_distances.max() > 0:
        feature_distances = feature_distances / feature_distances.max()
    
    # Create positive and negative masks
    # Positive pairs: Users with similar demographics (small feature distance)
    # Negative pairs: Users with different demographics (large feature distance)
    threshold = 0.3  # Threshold for considering demographics similar/different
    pos_mask = (feature_distances < threshold) & (torch.eye(batch_size, device=embeddings.device) == 0)
    neg_mask = (feature_distances > threshold)
    
    # For each user, we want:
    # 1. Similar embeddings for demographically similar users (positive pairs)
    # 2. Different embeddings for demographically different users (negative pairs)
    
    # Positive loss: -log(exp(sim(i,j)) / sum_k(exp(sim(i,k))))
    # This encourages similar embeddings for positive pairs
    pos_loss = 0
    if pos_mask.sum() > 0:
        # For numerical stability, subtract max similarity
        logits_max, _ = torch.max(similarities, dim=1, keepdim=True)
        logits = similarities - logits_max.detach()
        
        # Compute log_prob for positive pairs
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Mean over positive pairs
        pos_loss = -(pos_mask * log_prob).sum() / pos_mask.sum()
    
    # Negative loss: Push apart embeddings for negative pairs
    neg_loss = 0
    if neg_mask.sum() > 0:
        # We want to minimize similarity for negative pairs
        neg_loss = (neg_mask * torch.exp(similarities)).sum() / neg_mask.sum()
    
    # Combined loss
    loss = pos_loss + neg_loss
    
    return loss

def hybrid_movie_embedding_loss(movie_embeddings, user_embeddings, ratings, 
                               negative_user_embeddings=None, alpha=0.4, beta=0.2):
    """
    Hybrid loss function optimized for movie embeddings
    
    Args:
        movie_embeddings: Movie embeddings being trained
        user_embeddings: User embeddings (fixed) who rated these movies
        ratings: Actual ratings given by users
        negative_user_embeddings: Users who didn't rate these movies (for contrast)
        alpha: Weight for the ranking component
        beta: Weight for the content similarity component
    
    Returns:
        Combined loss, rating loss, ranking loss, and content loss
    """
    # Normalize embeddings
    movie_norm = nn.functional.normalize(movie_embeddings, p=2, dim=1)
    user_norm = nn.functional.normalize(user_embeddings, p=2, dim=1)
    
    # 1. Rating prediction component (MSE)
    similarity = torch.sum(movie_norm * user_norm, dim=1)
    predicted_ratings = 1 + 4 * (similarity + 1) / 2  # Scale to 1-5
    rating_loss = nn.MSELoss()(predicted_ratings, ratings)
    
    # 2. Ranking component (if negative samples provided)
    ranking_loss = torch.tensor(0.0, device=movie_embeddings.device)
    if negative_user_embeddings is not None:
        neg_norm = nn.functional.normalize(negative_user_embeddings, p=2, dim=1)
        
        # For each movie, we want users who rated it to be more similar than those who didn't
        pos_scores = similarity
        neg_scores = torch.matmul(movie_norm, neg_norm.t())
        
        # BPR-inspired loss: maximize difference between positive and negative scores
        diff = pos_scores.unsqueeze(1) - neg_scores
        ranking_loss = -torch.mean(torch.log(torch.sigmoid(diff)))
    
    # 3. Content similarity component (optional)
    # This encourages similar movies to have similar embeddings
    content_loss = torch.tensor(0.0, device=movie_embeddings.device)
    
    # Combined loss
    total_loss = rating_loss + alpha * ranking_loss + beta * content_loss
    
    return total_loss, rating_loss, ranking_loss, content_loss

def sample_negative_users(positive_users, all_users, k=5):
    """
    Sample users who didn't rate the movies in the current batch
    
    Args:
        positive_users: User embeddings who rated the movies
        all_users: All available user embeddings
        k: Number of negative samples per positive user
    
    Returns:
        Tensor of negative user embeddings
    """
    batch_size = positive_users.shape[0]
    all_users_count = all_users.shape[0]
    
    # For simplicity, just randomly sample from all users
    # In a production system, you'd want to ensure these users didn't actually rate the movies
    negative_indices = torch.randint(0, all_users_count, (batch_size, k), device=positive_users.device)
    
    # Reshape to (batch_size * k, embedding_dim)
    negative_users = all_users[negative_indices.flatten()]
    
    return negative_users

def main():
    parser = argparse.ArgumentParser(description='Train recommendation system with iterative refinement')
    parser.add_argument('--iterations', type=int, default=20, help='Number of refinement iterations')
    args = parser.parse_args()
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Create initial embeddings if not skipped
    create_initial_user_embeddings()
    create_initial_movie_embeddings()
    
    # First iteration: train with initial embeddings
    print("\n=== Starting Iteration 0 ===")
    train_user_embeddings(0, "model_initial_movies")
    train_movie_embeddings(0, "model_initial_users")
    
    # Subsequent iterations
    for i in range(1, args.iterations):
        print(f"\n=== Starting Iteration {i} ===")
        
        # Train user embeddings using movies from previous iteration
        prev_movie_collection = "movies" if i == 1 else f"movies_iter{i-1}"
        train_user_embeddings(i, prev_movie_collection)
        
        # Train movie embeddings using users from current iteration
        current_user_collection = f"users_iter{i}"
        train_movie_embeddings(i, current_user_collection)
    
    # Copy final embeddings to the main collections
    final_iteration = args.iterations - 1
    if final_iteration > 0:
        print("\n=== Copying Final Embeddings and Models ===")
        copy_collection(f"users_iter{final_iteration}", "users_final")
        copy_collection(f"movies_iter{final_iteration}", "movies_final")
        copy_final_models(final_iteration)
    else:
        copy_collection("users", "users_final")
        copy_collection("movies", "movies_final")
        copy_final_models(0)
    
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main()
