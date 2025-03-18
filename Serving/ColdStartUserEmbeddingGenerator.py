import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DataProcessing import DataProcessing
from VectorDataBase.DBHandler import MilvusHandler
from Training.ModelTrainSystem import EmbeddingProjector
from Embedding.UserEmbedding import featureExtract


def generate_newuser_embeddings(model_path, user_data):
    user_dict = user_data.generate_user_dict()
     
    user_data = pd.DataFrame([user_dict])
     
    user_embedding_features = featureExtract(user_data)

    input_dim = user_embedding_features.shape[1]
    print(f"input_dim: {input_dim}")
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