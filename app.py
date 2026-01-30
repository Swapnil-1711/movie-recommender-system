import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
from collections import Counter

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

# -------------------- CONFIG --------------------
TMDB_API_KEY = "f150a96d2baf9f4fbee4f80c73645d8f"
PLACEHOLDER_POSTER = "assets/universal_movie_poster.png"

# -------------------- FETCH POSTER (TMDB v3 CORRECT) --------------------
@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
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
        return None

    except requests.exceptions.RequestException:
        return None


# -------------------- TEXT → VECTOR --------------------
def text_to_vector(text):
    return Counter(text.split())


# -------------------- COSINE SIMILARITY --------------------
def cosine_similarity_manual(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[x] * vec2[x] for x in intersection)

    sum1 = sum(v ** 2 for v in vec1.values())
    sum2 = sum(v ** 2 for v in vec2.values())
    denominator = (sum1 ** 0.5) * (sum2 ** 0.5)

    if denominator == 0:
        return 0.0
    return numerator / denominator


# -------------------- LOAD DATA --------------------
movies_dict = pickle.load(open("movie_dict.pkl", "rb"))
movies = pd.DataFrame(movies_dict)


# -------------------- CACHE TAG VECTORS --------------------
@st.cache_data(show_spinner=False)
def get_vectors(tags_series):
    return tags_series.apply(text_to_vector).tolist()


# -------------------- RECOMMEND FUNCTION --------------------
def recommend(movie):
    vectors = get_vectors(movies["tags"])

    index = movies[movies["title"] == movie].index[0]
    target_vec = vectors[index]

    scores = []
    for i, vec in enumerate(vectors):
        if i == index:
            continue
        score = cosine_similarity_manual(target_vec, vec)
        scores.append((i, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:5]

    recommended_movies = []
    recommended_posters = []

    for i, _ in scores:
        movie_id = movies.iloc[i].movie_id
        recommended_movies.append(movies.iloc[i].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters


# -------------------- UI --------------------
st.title("🎬 Movie Recommender System")

selected_movie = st.selectbox(
    "Select a movie:",
    movies["title"].values
)

if st.button("Recommend"):
    with st.spinner("🎥 Finding movies for you..."):
        names, posters = recommend(selected_movie)

    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            st.text(names[i])
            if posters[i]:
                st.image(posters[i])
            else:
                st.image(PLACEHOLDER_POSTER)
