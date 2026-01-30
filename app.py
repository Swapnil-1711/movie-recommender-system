import streamlit as st
import pickle
import pandas as pd
import requests

# -------------------- CONFIG --------------------
TMDB_API_KEY = "27504f2c300a9b6057c468ab2bbd75c3"
PLACEHOLDER_POSTER = "assets/universal_movie_poster.png"

# -------------------- FETCH POSTER --------------------
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
        else:
            return None   # IMPORTANT: return None if no poster

    except requests.exceptions.RequestException:
        return None

# -------------------- RECOMMEND FUNCTION --------------------
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_movies_posters = []

    for i in movies_list:
        # IMPORTANT: movie_id must be TMDB ID
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_movies_posters

# -------------------- LOAD DATA --------------------
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

# -------------------- UI --------------------
st.title('🎬 Movie Recommender System')

selected_movie_name = st.selectbox(
    'Recommend:',
    movies['title'].values
)

if st.button('Recommend'):
    with st.spinner("🎬 Loading movies..."):
        names, posters = recommend(selected_movie_name)

    cols = st.columns(5)

    for i, col in enumerate(cols):
        with col:
            st.text(names[i])
            if posters[i]:
                st.image(posters[i])
            else:
                st.image("assets/universal_movie_poster.png")
