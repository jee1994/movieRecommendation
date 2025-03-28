package services

import (
	"context"
	"fmt"
	"log"

	"recommendation-service/Serving/Go/config"
	"recommendation-service/Serving/Go/model"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// VectorSearchService handles vector search operations
type VectorSearchService struct {
	config          *config.Config
	milvusClient    client.Client
	userCollection  string
	movieCollection string
}

// NewVectorSearchService creates a new vector search service
func NewVectorSearchService(cfg *config.Config) *VectorSearchService {
	// Connect to Milvus
	ctx := context.Background()
	c, err := client.NewGrpcClient(ctx, fmt.Sprintf("%s:%d", cfg.MilvusHost, cfg.MilvusPort))
	if err != nil {
		log.Fatalf("Failed to connect to Milvus: %v", err)
	}

	return &VectorSearchService{
		config:          cfg,
		milvusClient:    c,
		userCollection:  cfg.UserCollection,
		movieCollection: cfg.MovieCollection,
	}
}

// GetUserEmbedding retrieves a user's embedding from Milvus
func (s *VectorSearchService) GetUserEmbedding(userID int) ([]float32, error) {
	ctx := context.Background()

	// Prepare the query
	expr := fmt.Sprintf("UserID == %d", userID)

	// Execute the query
	queryResult, err := s.milvusClient.Query(
		ctx,
		s.userCollection,
		[]string{}, // Use empty partition
		expr,
		[]string{"UserID", "embedding"}, // Output fields
	)
	if err != nil {
		return nil, fmt.Errorf("failed to query Milvus: %w", err)
	}

	// Check if user exists
	if len(queryResult) == 0 || queryResult[0].Len() == 0 {
		return nil, nil // User not found
	}

	// Extract embedding from result
	embeddingField, ok := queryResult[1].(*entity.ColumnFloatVector)
	if !ok {
		return nil, fmt.Errorf("user found but embedding is empty")
	}
	data := embeddingField.Data()
	return data[0], nil
}

// SearchMovies searches for similar movies using an embedding vector
func (s *VectorSearchService) SearchMovies(embedding []float32, topK int) ([]model.Recommendation, error) {
	ctx := context.Background()

	// Convert embedding to Vector type
	vectors := []entity.Vector{entity.FloatVector(embedding)}

	sp, err := entity.NewIndexIvfFlatSearchParam(10)

	if err != nil {
		return nil, fmt.Errorf("failed to create search params: %w", err)
	}

	// Execute the search with function-based approach
	searchResult, err := s.milvusClient.Search(
		ctx,                          // Context
		s.movieCollection,            // Collection name
		[]string{},                   // Partitions (empty for all)
		"",                           // Expression (empty for no filter)
		[]string{"MovieID", "Title"}, // Output fields
		vectors,                      // Query vectors
		"embedding",                  // Vector field name
		entity.COSINE,                // Metric type
		topK,                         // Top K results
		sp,                           // Search parameters (nil for defaults)
	)

	if err != nil {
		return nil, fmt.Errorf("failed to search Milvus: %w", err)
	}

	// Process search results
	recommendations := make([]model.Recommendation, 0, topK)
	for _, result := range searchResult {
		ids := result.IDs.(*entity.ColumnInt64)
		titles, ok := result.Fields[1].(*entity.ColumnVarChar)

		if ok {
			for i := 0; i < ids.Len(); i++ {
				recommendations = append(recommendations, model.Recommendation{
					MovieID: int(ids.Data()[i]),
					Title:   titles.Data()[i],
					Score:   result.Scores[i],
				})
			}
		}
	}

	return recommendations, nil
}
