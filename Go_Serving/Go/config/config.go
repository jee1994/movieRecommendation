package config

import (
	"log"
	"os"
	"path/filepath"
	"strconv"
)

// Config holds all configuration for the service
type Config struct {
	MilvusHost             string
	MilvusPort             int
	UserCollection         string
	MovieCollection        string
	EmbeddingDim           int
	ModelPath              string
	ColdStartModelPath     string
	EmbeddingServerAddress string
}

// LoadConfig loads configuration from environment variables
func LoadConfig() *Config {
	milvusPort, _ := strconv.Atoi(getEnv("MILVUS_PORT", "19530"))
	embeddingDim, _ := strconv.Atoi(getEnv("EMBEDDING_DIM", "768"))
	currentDir, err := os.Getwd()
	if err != nil {
		log.Printf("Warning: couldn't get working directory: %v", err)
		currentDir = "."
	}
	projectRoot := filepath.Dir(filepath.Dir(currentDir))

	// Set default model path as absolute path
	defaultModelPath := filepath.Join(projectRoot, "models", "user_model_final.pt")
	return &Config{
		MilvusHost:             getEnv("MILVUS_HOST", "127.0.0.1"),
		MilvusPort:             milvusPort,
		UserCollection:         getEnv("USER_COLLECTION", "users_final"),
		MovieCollection:        getEnv("MOVIE_COLLECTION", "movies_final"),
		EmbeddingDim:           embeddingDim,
		ModelPath:              getEnv("MODEL_PATH", "models/user_model_final.pt"),
		ColdStartModelPath:     getEnv("COLD_START_MODEL_PATH", defaultModelPath),
		EmbeddingServerAddress: getEnv("EMBEDDING_SERVER_ADDRESS", "127.0.0.1:50052"),
	}
}

// Helper function to get environment variable with a default value
func getEnv(key, defaultValue string) string {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	return value
}
