package services

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"

	"recommendation-service/Serving/Go/config"
	"recommendation-service/Serving/Go/model"
)

// ColdStartGenerator generates embeddings for cold start users
type ColdStartGenerator struct {
	config     *config.Config
	modelPath  string
	pythonPath string
}

// NewColdStartGenerator creates a new cold start generator
func NewColdStartGenerator(cfg *config.Config) *ColdStartGenerator {
	// Find Python executable
	pythonPath := os.Getenv("PYTHON_PATH")
	if pythonPath == "" {
		pythonPath = "python3" // Default to python3
	}

	return &ColdStartGenerator{
		config:     cfg,
		modelPath:  cfg.ColdStartModelPath,
		pythonPath: pythonPath,
	}
}

// GenerateEmbedding generates an embedding for a cold start user
func (g *ColdStartGenerator) GenerateEmbedding(user model.User) ([]float32, error) {
	// Create a temporary JSON file to pass user data
	tempDir, err := ioutil.TempDir("", "coldstart")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer os.RemoveAll(tempDir)

	// Create user data JSON
	userData := map[string]interface{}{
		"user_id":         user.UserID,
		"user_gender":     user.Gender,
		"user_age":        user.Age,
		"user_occupation": user.Occupation,
		"user_zipcode":    user.ZipCode,
	}

	userDataJSON, err := json.Marshal(userData)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal user data: %w", err)
	}

	userDataFile := filepath.Join(tempDir, "user_data.json")
	if err := ioutil.WriteFile(userDataFile, userDataJSON, 0644); err != nil {
		return nil, fmt.Errorf("failed to write user data file: %w", err)
	}

	// Create output file path
	outputFile := filepath.Join(tempDir, "embedding.json")

	// Get project root path
	rootDir := os.Getenv("PWD")

	parentDir := filepath.Dir(rootDir)

	// Path to Python script
	scriptPath := filepath.Join(parentDir, "ColdStartUserEmbeddingGenerator.py")
	fmt.Printf("%v", g.modelPath)
	// Run Python script to generate embedding
	cmd := exec.Command(
		g.pythonPath,
		scriptPath,
		userDataFile,
		g.modelPath,
		outputFile,
	)

	// Set working directory to project root
	cmd.Dir = parentDir
	cmd.Env = append(os.Environ(), "PYTHONPATH="+rootDir)
	// Capture output
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("Python script output: %s", string(output))
		return nil, fmt.Errorf("failed to run Python script: %w", err)
	}

	// Read the generated embedding
	embeddingJSON, err := ioutil.ReadFile(outputFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read embedding file: %w", err)
	}

	// Parse the embedding
	var embeddingFloat []float32
	var embeddingFloat64 []float64
	if err := json.Unmarshal(embeddingJSON, &embeddingFloat64); err != nil {
		return nil, fmt.Errorf("failed to parse embedding: %w", err)
	}

	// Convert float64 to float32
	embeddingFloat = make([]float32, len(embeddingFloat64))
	for i, v := range embeddingFloat64 {
		embeddingFloat[i] = float32(v)
	}

	return embeddingFloat, nil
}
