import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DataProcessing import DataProcessing
from VectorDataBase.DBHandler import MilvusHandler
from Training.ModelTrainSystem import EmbeddingProjector
from Embedding.UserEmbedding import featureExtract


def generate_newuser_embeddings(model_path, user_data):
    user_dict = user_data.generate_user_dict()
     
    user_data = pd.DataFrame([user_dict])
     
    user_embedding_features = featureExtract(user_data, fit_scaler=False)
    print(f"extract festures {user_embedding_features}")

    input_dim = user_embedding_features.shape[1]
    model = EmbeddingProjector(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    user_embeddings_tensor = torch.tensor(user_embedding_features, dtype=torch.float32)
    
    with torch.no_grad():
        user_embedding = model(user_embeddings_tensor)
        user_embedding = nn.functional.normalize(user_embedding, p=2, dim=1)

    final_user_embedding = user_embedding[0]
    return final_user_embedding

class User:
    def __init__(self, user_id, user_gender, user_age, user_occupation, user_zipcode):
        self.user_id = user_id
        self.user_gender = user_gender
        self.user_age = user_age
        self.user_occupation = user_occupation
        self.user_zipcode = user_zipcode

    def generate_user_dict(self):
        return {
            'UserID': self.user_id,
            'Gender': self.user_gender,
            'Age': self.user_age,
            'Occupation': self.user_occupation,
            'Zipcode': self.user_zipcode
        }
    
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python ColdStartUserEmbeddingGenerator.py user_data.json model_path output_file.json")
        sys.exit(1)
        
    user_data_file = sys.argv[1]
    model_path = sys.argv[2]
    output_file = sys.argv[3]
    
    # Load user data
    with open(user_data_file, 'r') as f:
        user_data = json.load(f)
    
    # Create user object
    user = User(
        user_id=user_data["user_id"],
        user_gender=user_data["user_gender"],
        user_age=user_data["user_age"],
        user_occupation=user_data["user_occupation"],
        user_zipcode=user_data["user_zipcode"]
    )
    
    # Generate embedding
    embedding = generate_newuser_embeddings(model_path, user)
    
    # Save embedding
    with open(output_file, 'w') as f:
        json.dump(embedding.tolist(), f)