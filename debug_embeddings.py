import torch
import numpy as np
import pandas as pd
from Embedding.UserEmbedding import featureExtract
from Training.ModelTrainSystem import EmbeddingProjector
from Serving.ColdStartUserEmbeddingGenerator import User

def debug_user_embeddings():
    """Test if different demographics produce different embeddings"""
    # Create test users with different demographics
    test_users = pd.DataFrame([
        {"UserID": 1001, "Gender": "M", "Age": 25, "Occupation": 4, "Zipcode": "90210"},
        {"UserID": 1002, "Gender": "F", "Age": 25, "Occupation": 4, "Zipcode": "90210"},
        {"UserID": 1003, "Gender": "M", "Age": 55, "Occupation": 4, "Zipcode": "90210"},
        {"UserID": 1004, "Gender": "F", "Age": 55, "Occupation": 4, "Zipcode": "90210"}
    ])
    
    # Extract features
    features = featureExtract(test_users)
    print(f"Feature shape: {features.shape}")
    print("Feature differences between users:")
    for i in range(len(test_users)):
        for j in range(i+1, len(test_users)):
            diff = np.linalg.norm(features[i] - features[j])
            print(f"  User {i+1} vs User {j+1}: {diff:.6f}")
    
    # Load the model
    model_path = "models/user_model_final.pt"
    input_dim = features.shape[1]
    model = EmbeddingProjector(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Generate embeddings
    features_tensor = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        embeddings = model(features_tensor)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Compare embeddings
    embeddings_np = embeddings.numpy()
    print("\nEmbedding differences between users:")
    for i in range(len(test_users)):
        for j in range(i+1, len(test_users)):
            diff = np.linalg.norm(embeddings_np[i] - embeddings_np[j])
            cos_sim = np.dot(embeddings_np[i], embeddings_np[j])
            print(f"  User {i+1} vs User {j+1}: Distance={diff:.6f}, Similarity={cos_sim:.6f}")
    
    # Print the first few dimensions of each embedding
    print("\nFirst 5 dimensions of each embedding:")
    for i in range(len(test_users)):
        print(f"  User {i+1}: {embeddings_np[i][:5]}")
    
    return features, embeddings_np

if __name__ == "__main__":
    debug_user_embeddings() 