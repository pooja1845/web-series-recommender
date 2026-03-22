


import streamlit as st
import pickle
import pandas as pd
import requests

import gdown
import os

# Replace with your actual FILE IDs
SERIES_URL = "https://drive.google.com/uc?id=1oBs6KwhteJnapF-Phsw8RGfvPRcKC8U4"
SIMILARITY_URL = "https://drive.google.com/uc?id=1GHsolmShqcHPSIW4oYzwz_8NiYqs6zTm"

# Download only if file not present
if not os.path.exists("series.pkl"):
    gdown.download(SERIES_URL, "series.pkl", quiet=False)

if not os.path.exists("similarity.pkl"):
    gdown.download(SIMILARITY_URL, "similarity.pkl", quiet=False)

# ------------------- CONFIG -------------------
st.set_page_config(page_title="Web Series Recommender", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------- LOAD DATA -------------------
df = pickle.load(open("series.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))


# ------------------- OMDb API -------------------
API_KEY = "9dc171f0"

def fetch_details(series_name):
    try:
        url = f"http://www.omdbapi.com/?apikey={API_KEY}&t={series_name}"
        data = requests.get(url, timeout=5).json()

        if data['Response'] == 'True':
            return {
                "poster": data['Poster'],
                "rating": data['imdbRating'],
                "year": data['Year'],
                "genre": data['Genre']
            }
    except:
        pass

    return {
        "poster": "https://via.placeholder.com/300x450?text=No+Image",
        "rating": "N/A",
        "year": "N/A",
        "genre": "N/A"
    }

# ------------------- RECOMMEND FUNCTION -------------------
def recommend(series):
    series = series.lower()
    df['Series Title'] = df['Series Title'].str.lower()

    if series not in df['Series Title'].values:
        return ["Not Found"]

    index = df[df['Series Title'] == series].index[0]
    distances = similarity[index]

    series_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    return [df.iloc[i[0]]['Series Title'] for i in series_list]

# ------------------- CUSTOM STYLE (Optional Divider) -------------------
st.markdown(
    """
    <style>
    .vertical-divider {
        border-left: 2px solid #ccc;
        height: 500px;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------- MAIN LAYOUT -------------------
left_col, divider, right_col = st.columns([1, 0.05, 2])

# ------------------- LEFT SIDE -------------------
with left_col:
    st.header("🎬 Selected Series")

    selected_series = st.selectbox(
        "Select a Web Series",
        df['Series Title'].values
    )

    details = fetch_details(selected_series)

    st.image(details["poster"], width=200)
    st.write(f"⭐ Rating: {details['rating']}")
    st.write(f"📅 Year: {details['year']}")
    st.write(f"🎭 Genre: {details['genre']}")

# ------------------- DIVIDER -------------------
with divider:
    st.markdown('<div class="vertical-divider"></div>', unsafe_allow_html=True)

# ------------------- RIGHT SIDE -------------------
with right_col:
    st.header("🎯 Recommended Series")

    if st.button("Recommend"):
        recommendations = recommend(selected_series)

        cols = st.columns(5)

        for i in range(len(recommendations)):
            with cols[i]:
                rec = fetch_details(recommendations[i])
                st.image(rec["poster"],width=200)
                st.caption(recommendations[i])
                st.caption(f"⭐ {rec['rating']}")
