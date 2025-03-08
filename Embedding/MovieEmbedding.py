# pytorch_bert_embeddings.py
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DataProcessing import DataProcessing

def embedProcessedData():
    # Convert the Spark DataFrame to a Pandas DataFrame
    movies_pd = DataProcessing.processMovieData().toPandas()

    # Initialize BERT model and tokenizer (using 'bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    def get_text_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=64, padding='max_length')
        outputs = bert_model(**inputs)
        # Use the [CLS] token embedding as the sentence representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.squeeze().detach()

    # Create a text field by concatenating title and genres
    movies_pd['Text'] = movies_pd['Title'] + " " + movies_pd['Genres']

    # Generate BERT embeddings for each movie text
    bert_embeddings = movies_pd['Text'].apply(lambda x: get_text_embedding(x))
    # Stack embeddings into a tensor (each row is a movie embedding from BERT)
    bert_embeddings_tensor = torch.stack(bert_embeddings.tolist())

    # Prepare rating features: average rating and rating count
    avg_ratings = torch.tensor(movies_pd['AvgRating'].values, dtype=torch.float32).unsqueeze(1)
    rating_counts = torch.tensor(movies_pd['RatingCount'].values, dtype=torch.float32).unsqueeze(1)

    # Normalize the rating features
    avg_ratings_norm = (avg_ratings - avg_ratings.mean()) / avg_ratings.std()
    rating_counts_norm = (rating_counts - rating_counts.mean()) / rating_counts.std()

    # Concatenate BERT embedding with the normalized rating features
    # (Assume bert_embeddings_tensor shape: [num_movies, bert_dim])
    combined_features = torch.cat([bert_embeddings_tensor, avg_ratings_norm, rating_counts_norm], dim=1)

    # Define a simple PyTorch model to produce final movie embeddings
    class MovieEmbeddingNet(torch.nn.Module):
        def __init__(self, input_dim, output_dim=64):
            super(MovieEmbeddingNet, self).__init__()
            self.fc = torch.nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            return self.fc(x)

    input_dim = combined_features.shape[1]
    net = MovieEmbeddingNet(input_dim=input_dim, output_dim=64)

    # Generate final movie embeddings
    final_embeddings = net(combined_features)
    # Normalize the embeddings (L2 normalization)
    final_embeddings = torch.nn.functional.normalize(final_embeddings, p=2, dim=1)

    return (movies_pd, final_embeddings)

    # final_embeddings is a [num_movies x 64] tensor ready to be stored in Milvus
