# pytorch_user_embeddings.py
import torch
import torch.nn as nn
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DataProcessing import DataProcessing

def embedUserData():
    # Convert the Spark users DataFrame to Pandas
    users_pd = DataProcessing.processUserData().toPandas()

    # Encode categorical features: Gender, Occupation, and Zipcode
    le_gender = LabelEncoder()
    users_pd["GenderEncoded"] = le_gender.fit_transform(users_pd["Gender"])

    le_occupation = LabelEncoder()
    users_pd["OccupationEncoded"] = le_occupation.fit_transform(users_pd["Occupation"])

    le_zipcode = LabelEncoder()
    users_pd["ZipcodeEncoded"] = le_zipcode.fit_transform(users_pd["Zipcode"])

    # Create a user feature matrix (e.g., Gender, Age, Occupation, Zipcode)
    user_features = users_pd[["GenderEncoded", "Age", "OccupationEncoded", "ZipcodeEncoded"]].values
    user_features_tensor = torch.tensor(user_features, dtype=torch.float32)

    # Define a simple network for user embeddings
    class UserEmbeddingNet(nn.Module):
        def __init__(self, input_dim, embedding_dim=64):
            super(UserEmbeddingNet, self).__init__()
            self.fc = nn.Linear(input_dim, embedding_dim)
        
        def forward(self, x):
            return self.fc(x)

    user_net = UserEmbeddingNet(input_dim=user_features_tensor.shape[1], embedding_dim=64)
    user_embeddings = user_net(user_features_tensor)
    user_embeddings = torch.nn.functional.normalize(user_embeddings, p=2, dim=1)

    return users_pd, user_embeddings
    # user_embeddings: tensor of shape [num_users, 32]
