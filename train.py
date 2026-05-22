import pandas as pd
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# LOAD DATASET
# ==========================================

movies = pd.read_csv(
    "tmdb_5000_movies.csv"
)

# ==========================================
# KEEP IMPORTANT COLUMNS
# ==========================================

movies = movies[
    ["title", "overview"]
]

# ==========================================
# REMOVE NULL VALUES
# ==========================================

movies.dropna(inplace=True)

# ==========================================
# REDUCE DATASET SIZE
# (IMPORTANT FOR RENDER FREE PLAN)
# ==========================================

movies = movies.head(1000)

# ==========================================
# TEXT VECTORIZATION
# ==========================================

cv = CountVectorizer(
    max_features=2000,
    stop_words="english"
)

vectors = cv.fit_transform(
    movies["overview"]
).toarray()

# ==========================================
# COSINE SIMILARITY
# ==========================================

similarity = cosine_similarity(
    vectors
)

# ==========================================
# SAVE MODEL FILES
# ==========================================

pickle.dump(
    movies,
    open("movies.pkl", "wb")
)

pickle.dump(
    similarity,
    open("similarity.pkl", "wb")
)

print("Training completed successfully")
