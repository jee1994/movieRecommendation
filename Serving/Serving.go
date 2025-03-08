// main.go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
)

const (
	milvusAddr     = "localhost:19530"
	collectionName = "movies"
)

// searchSimilarMovies takes a query embedding and retrieves similar movies from Milvus.
func searchSimilarMovies(queryEmbedding []float32, topK int) {
	ctx := context.Background()

	// Connect to Milvus
	milvusClient, err := client.NewGrpcClient(ctx, milvusAddr)
	if err != nil {
		log.Fatalf("Failed to connect to Milvus: %v", err)
	}
	defer milvusClient.Close()

	// Define search parameters
	searchParams := map[string]interface{}{
		"metric_type": "L2",
		"params":      map[string]int{"nprobe": 10},
	}

	// Execute search in Milvus: here we search the "movies" collection using the query embedding.
	results, err := milvusClient.Search(ctx, collectionName, nil, "",
		[]string{"MovieID"},
		[]client.FloatVector{queryEmbedding},
		int64(topK), searchParams)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Println("Recommended Movies:")
	for _, result := range results {
		for _, id := range result.IDs {
			fmt.Printf("Movie ID: %v\n", id)
		}
	}
}

func main() {
	// Example query embedding (should be generated via the PyTorch+BERT pipeline)
	exampleEmbedding := []float32{
		0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
		0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
		0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
		0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5,
		0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4,
		0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3,
		0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2,
		0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1,
	} // This should be a 64-dimensional vector.
	searchSimilarMovies(exampleEmbedding, 5)
}
