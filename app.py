"""
Grocery Recommendation App

This Streamlit application provides a user interface for searching and recommending grocery products
based on various criteria such as store, pricing, and nutritional tags.

The app uses a product recommender system implemented in src/recommender.py to generate
personalized product recommendations based on user input.

Usage:
    Run this script using Streamlit:
    $ streamlit run app.py

Dependencies:
    - streamlit
    - pandas
    - src.recommender (custom module)

Author: [Ziba Atak]
Date: [22/9/2024]
Version: 2.0
"""

import streamlit as st
from src.recommender import load_data, ProductRecommender, save_search_history

def main():
    """
    Main function to run the Streamlit application.
    
    This function sets up the page configuration, loads the data, initializes the recommender,
    and handles user interactions for product search and recommendation display.
    """
    # Set up page configuration
    st.set_page_config(page_title="Grocery Recommendation App", page_icon="🛒", layout="wide")

    # Apply custom CSS styling
    st.markdown("""
    <style>
    .title {
        font-family: 'Montserrat Bold', sans-serif;
        font-size: 36px;
    }
    body {
        font-family: 'Montserrat Classic', sans-serif;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🛒 Grocery Recommendation App")

    # Load data and initialize recommender
    df = load_data()
    if df.empty:
        return

    recommender = ProductRecommender(df)
    search_history = []

    # Create search form
    with st.form("search_form"):
        col1, col2 = st.columns(2)
        with col1:
            search_query = st.text_input("Search:", key="search_query")
            top_n = st.number_input("Number of recommendations:", min_value=1, max_value=30, value=5, key="top_n")
        with col2:
            store = st.selectbox("By Store (optional):", [None] + sorted(df['STORE_NAME'].unique().tolist()), key="store")
            pricing = st.selectbox("By Price Level (optional):", [None, "Budget", "Premium", "Mid-range"], key="pricing")
            nutritional_tags = st.selectbox("By nutritional tags (optional):", 
                                            [None, 'Source of Protein', 'High Protein', 'Low Sugars', 'Low Sodium', 'Low Fat', 'Fat Free'],
                                            key="nutritional_tags")
        submitted = st.form_submit_button("Search")

    # Handle search submission
    if submitted:
        recommendations = recommender.search_products(search_query, top_n, store, pricing, nutritional_tags)
        search_history.append((search_query, recommendations))

        if recommendations.empty:
            st.warning("No products matching the filters.")
        else:
            st.subheader("Recommendations")
            for _, row in recommendations.iterrows():
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.image(row['IMAGE_URL'], width=150)
                    with col2:
                        st.markdown(f"**[{row['PRODUCT_NAME_T']}]({row['PRODUCT_LINK']})**")
                        st.write(f"Price: {row['PRODUCT_PRICE']}")
                        st.write(f"Brand: {row['PRODUCT_BRAND']}")
                        stores = df[df['PRODUCT_NAME_T'] == row['PRODUCT_NAME_T']]['STORE_NAME'].unique()
                        st.write(f"Stores: {', '.join(stores)}")
                        st.write(f"Pricing Category: {row['PRICE_TAGS']}")
                st.markdown("---")

        # Save search history
        save_search_history(search_history)

if __name__ == "__main__":
    main()