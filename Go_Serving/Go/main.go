package main

import (
	"log"
	"net/http"
	"os"

	"recommendation-service/Serving/Go/config"
	"recommendation-service/Serving/Go/services"

	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
)

func main() {
	// Load environment variables
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found, using environment variables")
	}

	// Initialize configuration
	cfg := config.LoadConfig()

	// Set up Gin router
	router := gin.Default()

	// Enable CORS
	router.Use(func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Origin, Content-Type, Content-Length, Accept-Encoding, Authorization")
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		c.Next()
	})

	// Initialize handlers
	h := services.NewHandler(cfg)

	// API routes
	v1 := router.Group("/api/v1")
	{
		// Health check
		v1.GET("/health", func(c *gin.Context) {
			c.JSON(http.StatusOK, gin.H{"status": "ok"})
		})

		// Recommendation endpoints
		v1.GET("/recommendations/user/:userId", h.GetRecommendationsForUser)
		v1.POST("/recommendations/cold-start", h.GetRecommendationsForColdStartUser)
		v1.POST("/recommendations/embedding", h.GetRecommendationsFromEmbedding)
	}

	// Get port from environment or use default
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// Start server
	log.Printf("Starting server on port %s", port)
	if err := router.Run(":" + port); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
