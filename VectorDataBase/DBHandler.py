# milvus_storage.py
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import numpy as np
import Embedding

# Get Final Embedding Vector 

movies_pd, final_embeddings = Embedding.embedProcessedData()

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define the schema for the Milvus collection
fields = [
    FieldSchema(name="MovieID", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=64)
]
schema = CollectionSchema(fields, "Movie embeddings collection")
collection = Collection("movies", schema)

# Prepare the data to insert:
# Convert final_embeddings tensor to a list of lists
movie_ids = movies_pd['MovieID'].tolist()
embeddings_list = final_embeddings.detach().cpu().numpy().tolist()

entities = [
    movie_ids,
    embeddings_list
]

# Insert the data into Milvus
collection.insert(entities)

# Create an index for fast similarity search
index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 100}}
collection.create_index(field_name="embedding", index_params=index_params)
