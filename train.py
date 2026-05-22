import pandas as pd
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv(
    "tmdb_5000_movies.csv"
)

# Keep useful columns
movies = movies[
    ["title", "overview"]
]

# Remove null values
movies.dropna(inplace=True)

# Convert text to vectors
cv = CountVectorizer(
    max_features=5000,
    stop_words="english"
)

vectors = cv.fit_transform(
    movies["overview"]
).toarray()

# Similarity matrix
similarity = cosine_similarity(
    vectors
)

# Save files
pickle.dump(
    movies,
    open("movies.pkl", "wb")
)

pickle.dump(
    similarity,
    open("similarity.pkl", "wb")
)

print("Training completed")
