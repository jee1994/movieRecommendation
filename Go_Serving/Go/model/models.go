package model

// User represents a user in the system
type User struct {
	UserID     int       `json:"userId"`
	Gender     string    `json:"gender"`
	Age        int       `json:"age"`
	Occupation int       `json:"occupation"`
	ZipCode    string    `json:"zipCode"`
	Embedding  []float32 `json:"embedding,omitempty"`
}

// Movie represents a movie in the system
type Movie struct {
	MovieID   int       `json:"movieId"`
	Title     string    `json:"title"`
	Genres    []string  `json:"genres,omitempty"`
	Year      int       `json:"year,omitempty"`
	Embedding []float32 `json:"embedding,omitempty"`
}

// Recommendation represents a movie recommendation with score
type Recommendation struct {
	MovieID int      `json:"movieId"`
	Title   string   `json:"title"`
	Score   float32  `json:"score"`
	Genres  []string `json:"genres,omitempty"`
}

// RecommendationResponse is the response for recommendation requests
type RecommendationResponse struct {
	Recommendations []Recommendation `json:"recommendations"`
	UserID          int              `json:"userId,omitempty"`
	Message         string           `json:"message,omitempty"`
}

// EmbeddingRequest represents a request with an embedding vector
type EmbeddingRequest struct {
	Embedding []float32 `json:"embedding"`
	TopK      int       `json:"topK,omitempty"`
}

// ColdStartUserRequest represents a request for a new user
type ColdStartUserRequest struct {
	User User `json:"user"`
	TopK int  `json:"topK,omitempty"`
}
