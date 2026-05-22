
from flask import Flask, request, jsonify, send_from_directory
import pickle
import requests
import os

from urllib.parse import quote

app = Flask(__name__)

# ==========================================
# TMDB API KEY
# ==========================================

API_KEY = "b0fc1b71e692d446ad0de8a81de937d6"

# ==========================================
# GENERATE MODEL FILES IF MISSING
# ==========================================

if not os.path.exists("movies.pkl"):

    import train

# ==========================================
# LOAD MODEL FILES
# ==========================================

movies = pickle.load(
    open("movies.pkl", "rb")
)

similarity = pickle.load(
    open("similarity.pkl", "rb")
)

# ==========================================
# POSTER CACHE
# ==========================================

poster_cache = {}

# ==========================================
# FETCH MOVIE POSTER
# ==========================================

def fetch_poster(movie_title):

    # Use cache for faster loading
    if movie_title in poster_cache:

        return poster_cache[movie_title]

    try:

        encoded_title = quote(movie_title)

        url = (
            "https://api.themoviedb.org/3/search/movie"
            f"?api_key={API_KEY}"
            f"&query={encoded_title}"
        )

        response = requests.get(
            url,
            timeout=5
        )

        data = response.json()

        results = data.get("results")

        # No results
        if not results:

            return None

        # Find first valid poster
        for movie in results:

            poster_path = movie.get(
                "poster_path"
            )

            if poster_path:

                poster_url = (
                    f"https://image.tmdb.org/t/p/w500"
                    f"{poster_path}"
                )

                # Save in cache
                poster_cache[movie_title] = poster_url

                return poster_url

        return None

    except Exception as e:

        print(e)

        return None

# ==========================================
# HOME ROUTE
# ==========================================

@app.route("/")
def home():

    return "Movie Recommendation System Running"

# ==========================================
# FRONTEND ROUTE
# ==========================================

@app.route("/ui")
def ui():

    return send_from_directory(
        ".",
        "index.html"
    )

# ==========================================
# RECOMMENDATION ROUTE
# ==========================================

@app.route("/recommend", methods=["POST"])
def recommend():

    data = request.json

    movie_name = data["movie"].lower()

    # Search movie
    matched_movies = movies[
        movies["title"]
        .str.lower()
        .str.contains(movie_name)
    ]

    # Movie not found
    if matched_movies.empty:

        return jsonify({
            "error": "Movie not found"
        })

    # First matched movie
    movie = matched_movies.iloc[0]["title"]

    # Movie index
    index = movies[
        movies["title"] == movie
    ].index[0]

    # Similarity scores
    distances = similarity[index]

    # Get recommended movies
    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:15]

    recommendations = []

    for i in movie_list:

        movie_data = movies.iloc[i[0]]

        poster = fetch_poster(
            movie_data.title
        )

        # Skip movies without posters
        if poster:

            recommendations.append({

                "title": movie_data.title,

                "poster": poster

            })

        # Limit to 5 movies
        if len(recommendations) == 5:

            break

    return jsonify(recommendations)

# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )
