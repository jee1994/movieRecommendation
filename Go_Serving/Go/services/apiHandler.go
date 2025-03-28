package services

import (
	"net/http"
	"strconv"

	"recommendation-service/Serving/Go/config"
	"recommendation-service/Serving/Go/model"

	"github.com/gin-gonic/gin"
)

// Handler handles HTTP requests
type Handler struct {
	recommender *RecommendationService
	config      *config.Config
}

// NewHandler creates a new handler
func NewHandler(cfg *config.Config) *Handler {
	recommender := NewRecommendationService(cfg)
	return &Handler{
		recommender: recommender,
		config:      cfg,
	}
}

// GetRecommendationsForUser handles requests for user recommendations
func (h *Handler) GetRecommendationsForUser(c *gin.Context) {
	// Parse user ID from path
	userIDStr := c.Param("userId")
	userID, err := strconv.Atoi(userIDStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid user ID"})
		return
	}

	// Parse top-k parameter
	topK := 5 // Default value
	topKStr := c.Query("topK")
	if topKStr != "" {
		parsedTopK, err := strconv.Atoi(topKStr)
		if err == nil && parsedTopK > 0 {
			topK = parsedTopK
		}
	}

	// Get recommendations
	recommendations, err := h.recommender.GetRecommendationsForUser(userID, topK)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Return response
	c.JSON(http.StatusOK, model.RecommendationResponse{
		Recommendations: recommendations,
		UserID:          userID,
	})
}

// GetRecommendationsForColdStartUser handles cold start user recommendations
func (h *Handler) GetRecommendationsForColdStartUser(c *gin.Context) {
	// Parse request body
	var request model.ColdStartUserRequest
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}

	// Set default topK if not provided
	if request.TopK <= 0 {
		request.TopK = 5
	}

	// Get recommendations for cold start user
	recommendations, err := h.recommender.GetRecommendationsForColdStartUser(request.User, request.TopK)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Return response
	c.JSON(http.StatusOK, model.RecommendationResponse{
		Recommendations: recommendations,
		UserID:          request.User.UserID,
	})
}

// GetRecommendationsFromEmbedding handles recommendations from raw embedding
func (h *Handler) GetRecommendationsFromEmbedding(c *gin.Context) {
	// Parse request body
	var request model.EmbeddingRequest
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}

	// Validate embedding dimensions
	if len(request.Embedding) != h.config.EmbeddingDim {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":    "Invalid embedding dimensions",
			"expected": h.config.EmbeddingDim,
			"received": len(request.Embedding),
		})
		return
	}

	// Set default topK if not provided
	if request.TopK <= 0 {
		request.TopK = 5
	}

	// Get recommendations from embedding
	recommendations, err := h.recommender.GetRecommendationsFromEmbedding(request.Embedding, request.TopK)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Return response
	c.JSON(http.StatusOK, model.RecommendationResponse{
		Recommendations: recommendations,
	})
}
