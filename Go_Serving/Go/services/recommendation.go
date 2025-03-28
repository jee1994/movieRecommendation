package services

import (
	"errors"
	"fmt"
	"log"

	"recommendation-service/Serving/Go/config"
	"recommendation-service/Serving/Go/model"
)

// RecommendationService provides movie recommendations
type RecommendationService struct {
	config       *config.Config
	vectorSearch *VectorSearchService
	coldStartGen *ColdStartGenerator
}

// NewRecommendationService creates a new recommendation service
func NewRecommendationService(cfg *config.Config) *RecommendationService {
	vectorSearch := NewVectorSearchService(cfg)
	coldStartGen := NewColdStartGenerator(cfg)

	return &RecommendationService{
		config:       cfg,
		vectorSearch: vectorSearch,
		coldStartGen: coldStartGen,
	}
}

// GetRecommendationsForUser gets recommendations for an existing user
func (s *RecommendationService) GetRecommendationsForUser(userID int, topK int) ([]model.Recommendation, error) {
	// Get user embedding from Milvus
	userEmbedding, err := s.vectorSearch.GetUserEmbedding(userID)
	if err != nil {
		return nil, fmt.Errorf("failed to get user embedding: %w", err)
	}

	// If user not found, return error
	if userEmbedding == nil {
		return nil, errors.New("user not found")
	}

	// Search for similar movies
	return s.vectorSearch.SearchMovies(userEmbedding, topK)
}

// GetRecommendationsForColdStartUser gets recommendations for a new user
func (s *RecommendationService) GetRecommendationsForColdStartUser(user model.User, topK int) ([]model.Recommendation, error) {
	// Generate embedding for cold start user
	embedding, err := s.coldStartGen.GenerateEmbedding(user)
	if err != nil {
		return nil, fmt.Errorf("failed to generate user embedding: %w", err)
	}

	// Log first few dimensions of the embedding
	if len(embedding) > 5 {
		log.Printf("Cold start user embedding (first 5 dims): [%.4f, %.4f, %.4f, %.4f, %.4f, ...]",
			embedding[0], embedding[1], embedding[2], embedding[3], embedding[4])
	}

	// Search for similar movies
	return s.vectorSearch.SearchMovies(embedding, topK)
}

// GetRecommendationsFromEmbedding gets recommendations from a raw embedding vector
func (s *RecommendationService) GetRecommendationsFromEmbedding(embedding []float32, topK int) ([]model.Recommendation, error) {
	// Search for similar movies
	return s.vectorSearch.SearchMovies(embedding, topK)
}
