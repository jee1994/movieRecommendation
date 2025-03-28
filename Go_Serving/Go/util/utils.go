package services

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
)

// SaveToJSON saves data to a JSON file
func SaveToJSON(data interface{}, filename string) error {
	// Create directory if it doesn't exist
	dir := filepath.Dir(filename)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Marshal data to JSON
	jsonData, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal data: %w", err)
	}

	// Write to file
	if err := ioutil.WriteFile(filename, jsonData, 0644); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}

// LoadFromJSON loads data from a JSON file
func LoadFromJSON(filename string, target interface{}) error {
	// Read file
	jsonData, err := ioutil.ReadFile(filename)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	// Unmarshal JSON
	if err := json.Unmarshal(jsonData, target); err != nil {
		return fmt.Errorf("failed to unmarshal data: %w", err)
	}

	return nil
}

// NormalizeVector normalizes a vector to unit length
func NormalizeVector(vector []float32) []float32 {
	// Calculate magnitude
	var sum float32
	for _, v := range vector {
		sum += v * v
	}
	magnitude := float32(0)
	if sum > 0 {
		magnitude = float32(1.0 / float32(sum))
	}

	// Normalize
	normalized := make([]float32, len(vector))
	for i, v := range vector {
		normalized[i] = v * magnitude
	}

	return normalized
}
