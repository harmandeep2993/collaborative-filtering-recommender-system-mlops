#frontend/app.py
"""
Streamlit frontend for movie recommendation system.
"""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("🎬 Movie Recommendation System")
st.write("Get personalized movie recommendations!")

user_id = st.number_input("Enter User ID", min_value=1, max_value=6040, value=1)
n = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)

if st.button("Get Recommendations"):
    with st.spinner("Getting recommendations..."):
        response = requests.post(
            f"{API_URL}/recommendations",
            json={"user_id": user_id, "n": n}
        )
        
        if response.status_code == 200:
            data = response.json()
            recommendations = data["recommendations"]
            st.subheader(f"Top {n} recommendations for User {user_id}:")
            
            for i, movie in enumerate(recommendations, 1):
                st.write(f"{i}. **{movie['title']}** (score: {movie['predicted_score']})")
        else:
            st.error(f"Error {response.status_code}: {response.text}")