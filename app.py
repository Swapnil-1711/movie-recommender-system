import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
import os
from collections import Counter

# -------------------- CONFIG --------------------
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
PLACEHOLDER_POSTER = "assets/universal_movie_poster.png"

# -------------------- FETCH POSTER --------------------
@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    if TMDB_API_KEY is None:
        return None

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US"
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get("poster_path")

        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path
        else:
            return None

    except requests.exceptions.RequestException:
        return None


# -------------------- TEXT → VECTOR --------------------
def text_to_vector(text):
    return Counter(text.split())


# -------------------- COSINE SIMILARITY (MANUAL) --------------------
def cosine_similarity_manual(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[x] * vec2[x] for x in intersection)

    sum1 = sum(v ** 2 for v in vec1.values())
    sum2 = sum(v ** 2 for v in vec2.values())
    denominator = (sum1 ** 0.5) * (sum2 ** 0.5)

    if denominator == 0:
        return 0.0
    return numerator / denominator


# -------------------- BUILD SIMILARITY MATRIX --------------------
@st.cache_data
def build_similarity(movies):
    vectors = movies["tags"].apply(text_to_vector).tolist()
    n = len(vectors)
    similarity = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            similarity[i][j] = cosine_similarity_manual(vectors[i], vectors[j])

    return similarity


# -------------------- LOAD DATA --------------------
movies_dict = pickle.load(open("movie_dict.pkl", "rb"))
movies = pd.DataFrame(movies_dict)

similarity = build_similarity(movies)


# -------------------- RECOMMEND FUNCTION --------------------
def recommend(movie):
    index = movies[movies["title"] == movie].index[0]
    distances = list(enumerate(similarity[index]))
    distances = sorted(distances, reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in distances:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters


# -------------------- UI --------------------
st.title("🎬 Movie Recommender System")

selected_movie_name = st.selectbox(
    "Select a movie:",
    movies["title"].values
)

if st.button("Recommend"):
    with st.spinner("🎥 Finding movies for you..."):
        names, posters = recommend(selected_movie_name)

    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            st.text(names[i])
            if posters[i]:
                st.image(posters[i])
            else:
                st.image(PLACEHOLDER_POSTER)
