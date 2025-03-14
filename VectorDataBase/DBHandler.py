# milvus_storage.py
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np
import os
import Embedding

class MilvusHandler:
    collection = None
    def __init__(self, host=None, port=None, collection_name=None, dim=None, dataType=None, needToReset=False):
        # Get connection details from environment variables or use defaults
        self.host = host or os.environ.get("MILVUS_HOST", "localhost")
        self.port = port or os.environ.get("MILVUS_PORT", "19530")
        self.collection_name = collection_name or "movies"
        self.dim = dim or 64
        self.dataType = dataType or "movie"
        self.connect()
        if needToReset:
            utility.drop_collection(self.collection_name)
        if self.dataType == "movie":
            self.create_movie_collection()
        elif self.dataType == "user":
            self.create_user_collection()
        self.load_collection()
        print("#### init ####", self.host, self.port, self.collection_name, self.dim)
        
    def connect(self):
        """Establish connection to Milvus server"""
        connections.connect("default", host=self.host, port=self.port)
        print(f"Connected to Milvus at {self.host}:{self.port}")
        
    def create_movie_collection(self):
        """Create a collection if it doesn't exist"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            print("#### collection already exists ####", self.collection)
            if len(self.collection.indexes) == 0:
                print("#### create index ####")
                self.create_index()
                print(f"Created index for collection {self.collection_name}")

            return self.collection
            
        # Define the schema for the Milvus collection
        fields = [
            FieldSchema(name="MovieID", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="Title", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        print("#### new collection")
        schema = CollectionSchema(fields, "Movie embeddings collection")
        self.collection = Collection(self.collection_name, schema)
        self.create_index()
        return self.collection
    
    def create_user_collection(self):
        """Create a collection if it doesn't exist"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            print("#### collection already exists ####", self.collection)
            if len(self.collection.indexes) == 0:
                print("#### create index ####")
                self.create_index()
                print(f"Created index for collection {self.collection_name}")

            return self.collection
        
        fields = [
            FieldSchema(name="UserID", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        print("#### new collection")
        schema = CollectionSchema(fields, "User embeddings collection")
        self.collection = Collection(self.collection_name, schema)
        self.create_index()
        return self.collection
        
    def insert_movie_data(self, movies_pd, embeddings):
        """Insert movie data and embeddings into Milvus"""
        # Convert embeddings tensor to a list of lists if needed
        if hasattr(embeddings, 'detach'):
            embeddings_list = embeddings.detach().cpu().numpy().tolist()
        else:
            embeddings_list = embeddings
            
        movie_ids = movies_pd['MovieID'].tolist()
        titles = movies_pd['Title'].tolist()
        
        entities = [
            movie_ids,
            titles,
            embeddings_list
        ]
        
        # Insert the data into Milvus
        insert_result = self.collection.insert(entities)
        print(f"Inserted {len(movie_ids)} movies into Milvus")
        return insert_result
    
    def insert_user_data(self, users_pd, embeddings):
        """Insert user data and embeddings into Milvus"""
        # Convert embeddings tensor to a list of lists if needed
        if hasattr(embeddings, 'detach'):
            embeddings_list = embeddings.detach().cpu().numpy().tolist()
        else:
            embeddings_list = embeddings
            
        user_ids = users_pd['UserID'].tolist()
        
        entities = [
            user_ids,
            embeddings_list
        ]
        
        # Insert the data into Milvus
        insert_result = self.collection.insert(entities)
        print(f"Inserted {len(user_ids)} users into Milvus")
        return insert_result
    
    def create_index(self, index_type="IVF_FLAT", metric_type="L2", nlist=100):
        """Create an index for fast similarity search"""
        print("#### create index ####")
        index_params = {"index_type": index_type, "metric_type": metric_type, "params": {"nlist": nlist}}
        self.collection.create_index(field_name="embedding", index_params=index_params)
        print(f"Created {index_type} index on embeddings")
        
    def load_collection(self):
        """Load collection into memory for search"""
        print("#### load collection ####", self.collection)
        self.collection.load()
        
    def search(self, query_embedding, top_k=10):
        """Search for similar movies"""
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        # Convert query to list format if it's a tensor
        if hasattr(query_embedding, 'detach'):
            query_list = query_embedding.detach().cpu().numpy().tolist()
        else:
            query_list = query_embedding
            
        results = self.collection.search(
            data=[query_list], 
            anns_field="embedding", 
            param=search_params,
            limit=top_k,
            output_fields=["MovieID"]
        )
        
        return results

    def get_entities(self, expr="id >= 0", output_fields=None, limit=100):
        """Get entities from collection based on expression"""
        try:
            if not hasattr(self, 'collection') or self.collection is None:
                self.create_collection()
            
            if self.collection is None:
                print("Failed to create collection")
                return []
            
            # Default output fields
            if output_fields is None:
                output_fields = ["*"]  # Get all fields
            
            # Query the collection
            results = self.collection.query(
                expr=expr,
                output_fields=output_fields,
                limit=limit
            )
            
            return results
        
        except Exception as e:
            print(f"Error getting entities: {e}")
            return []
    
    def get_all_movie_entities(self, limit=10000):
        """Get all movie entities from collection"""
        return self.get_entities(expr="MovieID >= 0", limit=limit)
    
    def get_all_user_entities(self, limit=10000):
        """Get all user entities from collection"""
        return self.get_entities(expr="UserID >= 0", limit=limit)