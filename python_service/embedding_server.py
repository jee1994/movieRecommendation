#!/usr/bin/env python3
import grpc
import time
import json
import logging
import argparse
import sys, os
from concurrent import futures
import numpy as np

# Import the generated proto classes
import embedding_service_pb2
import embedding_service_pb2_grpc

# Import your existing code
# Adjust the import path as needed
from Serving.ColdStartUserEmbeddingGenerator import generate_newuser_embeddings, User

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingServicer(embedding_service_pb2_grpc.EmbeddingServiceServicer):
    """Implementation of the EmbeddingService service."""
    
    def __init__(self, model_path):
        """Initialize the servicer with the model path."""
        self.model_path = model_path
        logger.info(f"Initializing embedding service with model path: {model_path}")
        
        # Optional: Load model here for better performance
        # This avoids loading it on each request
        # self.model = load_model(model_path)
    
    def GenerateUserEmbedding(self, request, context):
        """Generate embedding for a user.
        
        Args:
            request: The request containing user data
            context: The gRPC context
            
        Returns:
            An EmbeddingResponse containing the embedding vector
        """
        try:
            logger.info(f"Received request for user ID: {request.user_id}")
            
            # Create user object from request
            user = User(
                user_id=request.user_id,
                user_gender=request.gender,
                user_age=request.age,
                user_occupation=request.occupation,
                user_zipcode=request.zipcode
            )
            
            # Log user data for debugging (optional)
            logger.debug(f"Processing user: {vars(user)}")
            
            # Generate embedding
            embedding = generate_newuser_embeddings(self.model_path, user)
            
            # Convert to appropriate type if needed
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            # Create response
            response = embedding_service_pb2.EmbeddingResponse(
                embedding=embedding
            )
            
            logger.info(f"Successfully generated embedding for user {request.user_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            return embedding_service_pb2.EmbeddingResponse()

def serve(model_path, port=50052, max_workers=10):
    """Start the gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    embedding_service_pb2_grpc.add_EmbeddingServiceServicer_to_server(
        EmbeddingServicer(model_path), server
    )
    server_address = f'[::]:{port}'
    server.add_insecure_port(server_address)
    server.start()
    logger.info(f"Server started on port {port}")
    logger.info(f"Model path: {model_path}")
    
    try:
        # Keep the server running until interrupted
        while True:
            time.sleep(3600)  # Sleep for an hour
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(0)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the embedding gRPC server')
    parser.add_argument('--model-path', required=True, help='Path to the model file')
    parser.add_argument('--port', type=int, default=50052, help='Port to listen on')
    parser.add_argument('--max-workers', type=int, default=10, help='Maximum number of workers')
    
    args = parser.parse_args()
    
    # Start the server
    logger.info(f"Starting embedding server with model: {args.model_path}")
    serve(args.model_path, args.port, args.max_workers)