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
from pymilvus import utility

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

def create_initial_user_embeddings():
    """Create initial user embeddings based only on demographic features"""
    print("\n=== Creating Initial User Embeddings ===")
    
    # Load user data
    users_pd, _ = DataProcessing.processUserData()
    users_pd = users_pd.toPandas()
    print(f"Loaded {len(users_pd)} users")
    
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
    num_occupations = users_pd['Occupation'].max() + 1  # Assuming 0-based indexing
    
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
    
    # 4. Locality-sensitive hashing for Zipcode
    # Convert zipcode to numeric if it's not already
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
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(combined_features)
    
    # Convert to tensor
    initial_features = torch.tensor(normalized_features, dtype=torch.float32)
    
    # Create a simple projection to 768 dimensions
    input_dim = initial_features.shape[1]
    model = SimpleProjector(input_dim)
    
    # Project features to 768 dimensions
    with torch.no_grad():
        user_embeddings = model(initial_features)
        user_embeddings = nn.functional.normalize(user_embeddings, p=2, dim=1)
    
    print(f"Initial user embeddings shape: {user_embeddings.shape}")
    
    # Store in vector database
    print("#### insert initial user data ####")
    db_handler = MilvusHandler(host="localhost", port=19530, collection_name="initial_users", dim=768, dataType="user", needToReset=True)
    db_handler.insert_user_data(users_pd, user_embeddings)
    
    return users_pd, user_embeddings

def create_initial_movie_embeddings():
    """Create initial movie embeddings based only on BERT text features"""
    print("\n=== Creating Initial Movie Embeddings ===")
    
    # Load movie data
    movies_pd = DataProcessing.processMovieData().toPandas()
    print(f"Loaded {len(movies_pd)} movies")
    
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
    
    # Apply normalization
    movie_embeddings = nn.functional.normalize(bert_embeddings_tensor, p=2, dim=1)
    
    print(f"Initial movie embeddings shape: {movie_embeddings.shape}")
    
    # Store in vector database
    print("#### insert initial movie data ####")
    db_handler = MilvusHandler(host="localhost", port=19530, collection_name="initial_movies", dim=768, dataType="movie", needToReset=True)
    db_handler.insert_movie_data(movies_pd, movie_embeddings)
    
    return movies_pd, movie_embeddings

def train_user_embeddings(iteration, movie_collection):
    """Train user embeddings based on movie embeddings from the specified collection"""
    print(f"\n=== Training User Embeddings (Iteration {iteration}) ===")
    
    # Load user data
    users_pd, ratings_pd = DataProcessing.processUserData()
    users_pd = users_pd.toPandas()
    ratings_pd = ratings_pd.toPandas()
    print(f"Loaded {len(users_pd)} users")
    
    print(f"Loaded {len(ratings_pd)} ratings")
    
    # Extract user features (same as in create_initial_user_embeddings)
    # [Code omitted for brevity - same as in create_initial_user_embeddings]
    
    # 1. One-hot encoding for Gender
    gender_one_hot = pd.get_dummies(users_pd['Gender'], prefix='Gender')
    
    # 2. Bracket encoding for Age
    age_brackets = [0, 18, 25, 35, 45, 55, 65, 100]
    bracket_labels = ['0-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    users_pd['AgeBracket'] = pd.cut(users_pd['Age'], bins=age_brackets, labels=bracket_labels, right=False)
    age_bracket_one_hot = pd.get_dummies(users_pd['AgeBracket'], prefix='Age')
    
    # 3. Occupation embedding
    num_occupations = users_pd['Occupation'].max() + 1
    occupation_embedding_dim = min(10, num_occupations)
    np.random.seed(42)
    occupation_embedding_matrix = np.random.normal(0, 1, (num_occupations, occupation_embedding_dim))
    occupation_embeddings = np.array([
        occupation_embedding_matrix[int(code)] for code in users_pd['Occupation']
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
    
    # Combine features
    gender_features = gender_one_hot.values
    age_features = age_bracket_one_hot.values
    combined_features = np.hstack([
        gender_features,
        age_features,
        occupation_embeddings,
        zipcode_features_array
    ])
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(combined_features)
    initial_features = torch.tensor(normalized_features, dtype=torch.float32)
    
    # Load movie embeddings from the specified collection
    try:
        movie_db = MilvusHandler(host="localhost", port=19530, collection_name=movie_collection, dim=768, dataType="movie")
        movie_data = movie_db.get_all_movie_entities()
        
        # Create a mapping from MovieID to embedding
        movie_id_to_embedding = {}
        for movie in movie_data:
            movie_id_to_embedding[movie['MovieID']] = torch.tensor(movie['embedding'], dtype=torch.float32)
        
        print(f"Loaded {len(movie_id_to_embedding)} movie embeddings from {movie_collection}")
    except Exception as e:
        print(f"Error loading movie embeddings: {e}")
        return None, None
    
    # Create a mapping from UserID to index
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(users_pd['UserID'])}
    
    # Initialize the model
    input_dim = initial_features.shape[1]
    model = EmbeddingProjector(input_dim=input_dim)
    
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
        print(f"Epoch {batch_size} {len(train_data)}")
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
        print(f"Epoch {epoch+1} {num_batches}")
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Generate final embeddings for all users
    with torch.no_grad():
        user_embeddings = model(initial_features)
        user_embeddings = nn.functional.normalize(user_embeddings, p=2, dim=1)
    
    # Get the final dimension
    embedding_dim = user_embeddings.shape[1]
    print(f"Final user embeddings shape: {user_embeddings.shape}")
    
    # Store in vector database
    collection_name = f"users_iter{iteration}" if iteration > 0 else "users"
    print(f"#### insert user data to {collection_name} ####")
    db_handler = MilvusHandler(host="localhost", port=19530, collection_name=collection_name, dim=embedding_dim, dataType="user")
    db_handler.insert_user_data(users_pd, user_embeddings)
    
    return users_pd, user_embeddings

def train_movie_embeddings(iteration, user_collection):
    """Train movie embeddings based on user embeddings from the specified collection"""
    print(f"\n=== Training Movie Embeddings (Iteration {iteration}) ===")
    
    # Load movie data
    movies_pd = DataProcessing.processMovieData().toPandas()
    print(f"Loaded {len(movies_pd)} movies")
    
    # Load ratings data
    _, ratings_pd = DataProcessing.processUserData()
    ratings_pd = ratings_pd.toPandas()
    print(f"Loaded {len(ratings_pd)} ratings")
    
    # Load user embeddings from the specified collection
    try:
        user_db = MilvusHandler(host="localhost", port=19530, collection_name=user_collection, dim=768, dataType="user")
        user_data = user_db.get_all_user_entities()
        
        # Create a mapping from UserID to embedding
        user_id_to_embedding = {}
        for user in user_data:
            user_id_to_embedding[user['UserID']] = torch.tensor(user['embedding'], dtype=torch.float32)
        
        print(f"Loaded {len(user_id_to_embedding)} user embeddings from {user_collection}")
    except Exception as e:
        print(f"Error loading user embeddings: {e}")
        return None, None
    
    # Initialize BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    
    # Create a text field by concatenating title and genres
    movies_pd['Text'] = movies_pd['Title'] + " [SEP] " + movies_pd['Genres'].str.replace('|', ' ')
    
    # Generate BERT embeddings in batches
    batch_size = 8
    bert_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(movies_pd), batch_size), desc="Generating BERT embeddings"):
            batch_texts = movies_pd['Text'][i:i+batch_size].tolist()
            
            inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, 
                              max_length=512, padding='max_length')
            
            outputs = bert_model(**inputs)
            
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            bert_embeddings.append(cls_embeddings.detach().numpy())
    
    # Concatenate all batches
    bert_embeddings = np.vstack(bert_embeddings)
    
    # Convert to tensor
    bert_embeddings_tensor = torch.tensor(bert_embeddings, dtype=torch.float32)
    
    # Create a mapping from MovieID to index
    movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies_pd['MovieID'])}
    
    # Initialize the model
    input_dim = bert_embeddings_tensor.shape[1]  # BERT dimension (768)
    model = EmbeddingProjector(input_dim=input_dim)
    
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
    collection_name = f"movies_iter{iteration}" if iteration > 0 else "movies"
    print(f"#### insert movie data to {collection_name} ####")
    db_handler = MilvusHandler(host="localhost", port=19530, collection_name=collection_name, dim=embedding_dim, dataType="movie")
    db_handler.insert_movie_data(movies_pd, movie_embeddings)
    
    return movies_pd, movie_embeddings

def copy_collection(source, target):
    """Copy data from source collection to target collection"""

    if target == "users_final":
        try:
            # Connect to source collection
            source_db = MilvusHandler(host="localhost", port=19530, collection_name=source, dim=768, dataType="user")
            entities = source_db.get_all_user_entities()
            
            if not entities:
                print(f"No entities found in {source}")
                return False
            
            # Extract data and embeddings
            data_df = pd.DataFrame([{k: v for k, v in entity.items() if k != 'embedding'} for entity in entities])
            embeddings = torch.tensor(np.array([entity['embedding'] for entity in entities]))
            
            # Create target collection and insert data
            target_db = MilvusHandler(host="localhost", port=19530, collection_name=target, dim=768, dataType="user")
            target_db.insert_user_data(data_df, embeddings)
            
            print(f"Copied {len(entities)} entities from {source} to {target}")
            return True
        except Exception as e:
            print(f"Error copying collection: {e}")
            return False
    else:
        try:
            # Connect to source collection
            source_db = MilvusHandler(host="localhost", port=19530, collection_name=source, dim=768, dataType="movie")
            entities = source_db.get_all_movie_entities()
            
            if not entities:
                print(f"No entities found in {source}")
                return False
            
            # Extract data and embeddings
            data_df = pd.DataFrame([{k: v for k, v in entity.items() if k != 'embedding'} for entity in entities])
            embeddings = torch.tensor(np.array([entity['embedding'] for entity in entities]))
            
            # Create target collection and insert data
            target_db = MilvusHandler(host="localhost", port=19530, collection_name=target, dim=768, dataType="movie")
            target_db.insert_movie_data(data_df, embeddings)
            
            print(f"Copied {len(entities)} entities from {source} to {target}")
            return True
        except Exception as e:
            print(f"Error copying collection: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Train recommendation system with iterative refinement')
    parser.add_argument('--iterations', type=int, default=3, help='Number of refinement iterations')
    parser.add_argument('--skip-initial', action='store_true', help='Skip initial embedding creation')
    args = parser.parse_args()
    MilvusHandler.drop_all_collections()
    
    create_initial_user_embeddings()
    create_initial_movie_embeddings()
    
    # First iteration: train with initial embeddings
    print("\n=== Starting Iteration 0 ===")
    train_user_embeddings(0, "initial_movies")
    train_movie_embeddings(0, "initial_users")
    
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
        print("\n=== Copying Final Embeddings to Main Collections ===")
        copy_collection(f"users_iter{final_iteration}", "users_final")
        copy_collection(f"movies_iter{final_iteration}", "movies_final")
    
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main()
