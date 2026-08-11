import streamlit as st
import pandas as pd
import requests
import numpy as np
import textwrap
import concurrent.futures
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Web Series Recommender", layout="wide", initial_sidebar_state="collapsed")

# ------------------- STYLING & FONTS -------------------
css_styles = """
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #f7f5f0;
    color: #1c1917;
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 4.5rem;
    padding-bottom: 0.5rem;
}

.section-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    color: #451a03;
    margin-bottom: 8px;
    border-left: 4px solid #d97706;
    padding-left: 8px;
    display: flex;
    align-items: center;
}

.selected-card {
    background: #ffffff;
    border: 1px solid #e7e5e4;
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 4px 15px rgba(27, 26, 25, 0.03);
    display: flex;
    gap: 20px;
    margin-top: 10px;
    margin-bottom: 10px;
    height: 380px;
    box-sizing: border-box;
}

.selected-poster {
    width: 220px;
    height: 346px;
    border-radius: 8px;
    object-fit: cover;
    box-shadow: 0 4px 12px rgba(27, 26, 25, 0.1);
    border: 1px solid #d6d3d1;
}

.selected-info {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}

.selected-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #1c1917;
    margin: 0 0 8px 0;
    line-height: 1.2;
}

.selected-meta {
    display: flex;
    gap: 8px;
    font-size: 0.85rem;
    color: #44403c;
    margin-bottom: 8px;
    flex-wrap: wrap;
}

.selected-meta span {
    background: #f5f5f4;
    color: #44403c;
    padding: 2px 6px;
    border-radius: 6px;
    border: 1px solid #e7e5e4;
    font-weight: 500;
    font-size: 0.75rem;
}

.selected-plot {
    font-size: 0.85rem;
    color: #57534e;
    line-height: 1.5;
    margin: 8px 0 0 0;
    display: -webkit-box;
    -webkit-line-clamp: 8;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.movies-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 15px;
    width: 100%;
    margin-top: 10px;
}

.movie-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 10px;
    border: 1px solid #e7e5e4;
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    height: 440px;
    display: flex;
    flex-direction: column;
    box-sizing: border-box;
    box-shadow: 0 4px 8px rgba(27, 26, 25, 0.02);
}

.movie-card:hover {
    transform: translateY(-5px);
    border-color: #d97706;
    box-shadow: 0 8px 16px rgba(217, 119, 6, 0.12);
}

.movie-card img {
    width: 100%;
    border-radius: 8px;
    object-fit: cover;
    aspect-ratio: 2 / 3;
    height: auto;
    box-shadow: 0 2px 6px rgba(27, 26, 25, 0.06);
    border: 1px solid #e7e5e4;
}

.movie-card-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    color: #1c1917;
    margin: 8px 0 4px 0;
    font-size: 0.9rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    text-align: center;
}

.movie-card-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: #57534e;
    font-family: 'Inter', sans-serif;
    margin-bottom: 6px;
    padding: 0 2px;
}

.movie-card-rating {
    color: #d97706;
    font-weight: 600;
}

.movie-card-plot {
    font-size: 0.75rem;
    color: #57534e;
    line-height: 1.4;
    margin: 6px 0 8px 0;
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;
    text-align: left;
}

.movie-card-genre {
    color: #78716c;
    font-size: 0.7rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-top: auto;
    padding-top: 6px;
    border-top: 1px solid #f5f5f4;
    text-align: center;
}

div.stButton > button {
    background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
    color: #ffffff;
    font-family: 'Outfit', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    border: none;
    padding: 8px 16px;
    border-radius: 8px;
    box-shadow: 0 3px 10px rgba(217, 119, 6, 0.15);
    transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
    width: 100%;
    margin-top: 8px;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #ea580c 0%, #c2410c 100%);
    box-shadow: 0 6px 24px rgba(217, 119, 6, 0.3);
    transform: translateY(-2px);
    color: #ffffff;
    border: none;
}

div.stButton > button:active {
    transform: translateY(1px);
}

div[data-baseweb="select"] {
    border-radius: 10px !important;
    background-color: #ffffff !important;
    border: 1px solid #d6d3d1 !important;
}

div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: none !important;
}

div[data-baseweb="select"] div {
    color: #1c1917 !important;
}

div[data-baseweb="select"] input {
    color: #1c1917 !important;
}

@media (max-width: 1200px) {
    .movies-grid {
        grid-template-columns: repeat(3, 1fr);
    }
    .movie-card {
        height: auto;
        min-height: 440px;
    }
}

@media (max-width: 992px) {
    .selected-card {
        flex-direction: column;
        height: auto;
        align-items: center;
        text-align: center;
        padding: 16px;
    }
    
    .selected-poster {
        width: 140px;
        height: auto;
        aspect-ratio: 2 / 3;
        margin-bottom: 12px;
    }
}

@media (max-width: 768px) {
    .movies-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: 15px;
    }
    
    .movie-card {
        height: auto;
        min-height: 400px;
    }
}

@media (max-width: 480px) {
    .movies-grid {
        grid-template-columns: 1fr;
    }
    
    .movie-card {
        height: auto;
        min-height: auto;
    }
}
"""

font_links = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Outfit:wght@400;600;700;800&display=swap" rel="stylesheet">'
)

st.markdown(
    f'<meta name="referrer" content="no-referrer">{font_links}<style>{css_styles.replace("\r", "").replace("\n", " ")}</style>',
    unsafe_allow_html=True
)

# ------------------- LOAD DATA & ENGINE -------------------
@st.cache_resource
def load_recommender_system():
    # Read the local CSV
    df = pd.read_csv("IMDB_TV_Shows_Clean.csv")
    df['Series Title'] = df['Series Title'].fillna('Unknown')
    
    # Process for vectorization
    df_clean = df.copy()
    df_clean['Genre'] = df_clean['Genre'].fillna('')
    df_clean['Description'] = df_clean['Description'].fillna('')
    df_clean['IMDB Rating Clean'] = pd.to_numeric(df_clean['IMDB Rating'], errors='coerce').fillna(6.0)
    
    # Genre Vectorizer (Custom split by comma tokenizer)
    genre_vectorizer = TfidfVectorizer(analyzer=lambda s: [g.strip().lower() for g in s.split(',') if g.strip()])
    genre_matrix = genre_vectorizer.fit_transform(df_clean['Genre'])
    
    # Description Vectorizer
    desc_vectorizer = TfidfVectorizer(stop_words='english')
    desc_matrix = desc_vectorizer.fit_transform(df_clean['Description'])
    
    # Normalize ratings to [0, 1] range for soft boosting
    rating_scores = df_clean['IMDB Rating Clean'].values / 10.0
    
    return df, genre_matrix, desc_matrix, rating_scores

try:
    df, genre_matrix, desc_matrix, rating_scores = load_recommender_system()
except Exception as e:
    st.error(f"Error initializing recommender system: {e}")
    st.stop()

# ------------------- OMDb API FETCH -------------------
API_KEY = "9dc171f0"

@st.cache_data
def fetch_details(series_name):
    try:
        url = f"http://www.omdbapi.com/?apikey={API_KEY}&t={series_name}"
        data = requests.get(url, timeout=3).json()

        if data.get('Response') == 'True':
            return {
                "poster": data['Poster'] if data.get('Poster') and data['Poster'] != "N/A" else "https://via.placeholder.com/300x450?text=No+Image",
                "rating": data.get('imdbRating', 'N/A'),
                "year": data.get('Year', 'N/A'),
                "genre": data.get('Genre', 'N/A'),
                "plot": data.get('Plot', '')
            }
    except Exception:
        pass

    # Fallback to local CSV dataset
    poster = "https://via.placeholder.com/300x450?text=No+Image"
    rating = "N/A"
    year = "N/A"
    genre = "N/A"
    plot = ""
    
    matching = df[df['Series Title'] == series_name]
    if len(matching) > 0:
        local_row = matching.iloc[0]
        if pd.notna(local_row['IMDB Rating']):
            rating = str(local_row['IMDB Rating'])
        if pd.notna(local_row['Year Released']):
            year = str(local_row['Year Released'])
        if pd.notna(local_row['Genre']):
            genre = str(local_row['Genre'])
        if pd.notna(local_row['Description']):
            plot = str(local_row['Description'])
                
    return {
        "poster": poster,
        "rating": rating,
        "year": year,
        "genre": genre,
        "plot": plot
    }

# ------------------- RECOMMENDATION ENGINE -------------------
def recommend_series(series_title, w_genre=0.5, w_desc=0.3, w_rating=0.2):
    # Find matching index
    matching = df[df['Series Title'] == series_title]
    if len(matching) == 0:
        # Retry case-insensitive
        matching = df[df['Series Title'].str.lower() == series_title.lower()]
        if len(matching) == 0:
            return pd.DataFrame()
            
    idx = matching.index[0]
    
    # Calculate similarity on-the-fly
    genre_sim = cosine_similarity(genre_matrix[idx], genre_matrix).flatten()
    desc_sim = cosine_similarity(desc_matrix[idx], desc_matrix).flatten()
    
    # Composite score
    combined_scores = (w_genre * genre_sim) + (w_desc * desc_sim) + (w_rating * rating_scores)
    
    # Exclude the selected series from its own recommendations
    combined_scores[idx] = -1.0
    
    # Sort and pick top 5
    top_indices = np.argsort(combined_scores)[::-1][:5]
    
    return df.iloc[top_indices]

# ------------------- APPLICATION HEADER -------------------
st.markdown(
    '<h1 style="text-align: center; font-family: \'Outfit\', sans-serif; font-weight: 800; font-size: 2.2rem; color: #451a03; margin: 0 0 15px 0;">🎬 Web Series Recommender</h1>',
    unsafe_allow_html=True
)

# ------------------- LAYOUT SETUP -------------------
left_col, divider_col, right_col = st.columns([1.2, 0.05, 2.0])

# ------------------- LEFT SIDE (SELECTION & DETAILS) -------------------
with left_col:
    st.markdown('<div class="section-title">🔍 Selected Series</div>', unsafe_allow_html=True)
    
    selected_series = st.selectbox(
        "Choose a Web Series",
        df['Series Title'].values,
        index=0,
        label_visibility="collapsed"
    )
    
    # Fetch details
    details = fetch_details(selected_series)
    
    # Render Selected Show details in a premium card
    selected_card_html = (
        f'<div class="selected-card">'
        f'<img class="selected-poster" src="{details["poster"]}" />'
        f'<div class="selected-info">'
        f'<h2 class="selected-title">{selected_series}</h2>'
        f'<div class="selected-meta">'
        f'<span>⭐ Rating: {details["rating"]}</span>'
        f'<span>📅 Year: {details["year"]}</span>'
        f'</div>'
        f'<div style="font-size: 0.9rem; color: #c084fc; font-weight: 600; margin-bottom: 10px;">🎭 {details["genre"]}</div>'
        f'<p class="selected-plot">{details["plot"][:450] + ("..." if len(details["plot"]) > 450 else "")}</p>'
        f'</div>'
        f'</div>'
    )
    st.markdown(selected_card_html, unsafe_allow_html=True)
    
    # Large glowing button for trigger
    recommend_button = st.button("Generate Recommendations")

# ------------------- RIGHT SIDE (RECOMMENDATIONS) -------------------
with right_col:
    st.markdown('<div class="section-title">🎯 Recommended Series</div>', unsafe_allow_html=True)
    
    # Initialize session state variables if not present
    if 'recommended_for' not in st.session_state:
        st.session_state['recommended_for'] = None
    if 'recommendations_generated' not in st.session_state:
        st.session_state['recommendations_generated'] = False

    # If the user changed the selection, reset the generated flag
    if selected_series != st.session_state['recommended_for']:
        st.session_state['recommendations_generated'] = False

    # If button is clicked, set state to show recommendations
    if recommend_button:
        st.session_state['recommendations_generated'] = True
        st.session_state['recommended_for'] = selected_series

    # Render recommendations if generated
    if st.session_state['recommendations_generated']:
        recommendations = recommend_series(selected_series)
        
        if recommendations.empty:
            st.warning("No recommendations found.")
        else:
            cards_html = []
            
            # Fetch details in parallel
            titles = recommendations['Series Title'].values
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                rec_details_list = list(executor.map(fetch_details, titles))
            
            for i, title in enumerate(titles):
                rec_details = rec_details_list[i]
                
                # Extract clean year
                year_clean = rec_details['year'].split('–')[0]
                
                # Truncate plot for recommended series card
                plot_truncated = rec_details['plot'][:120] + ("..." if len(rec_details['plot']) > 120 else "") if rec_details['plot'] else "No description available."
                
                card_html = (
                    f'<div class="movie-card">'
                    f'<img src="{rec_details["poster"]}" />'
                    f'<div class="movie-card-title" title="{title}">{title}</div>'
                    f'<div class="movie-card-meta">'
                    f'<span class="movie-card-rating">⭐ {rec_details["rating"]}</span>'
                    f'<span>📅 {year_clean}</span>'
                    f'</div>'
                    f'<div class="movie-card-plot" title="{rec_details["plot"]}">{plot_truncated}</div>'
                    f'<div class="movie-card-genre" title="{rec_details["genre"]}">🎭 {rec_details["genre"]}</div>'
                    f'</div>'
                )
                cards_html.append(card_html)
                
            grid_html = f'<div class="movies-grid">{"".join(cards_html)}</div>'
            st.markdown(grid_html, unsafe_allow_html=True)
    else:
        # Show a beautiful placeholder explaining how to generate recommendations
        st.info("🎯 Click the 'Generate Recommendations' button on the left to see recommendations for the selected series.")
