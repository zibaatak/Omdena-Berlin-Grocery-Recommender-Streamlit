"""
Grocery Product Recommender Module

This module provides functionality for loading grocery product data, preprocessing text,
and recommending products based on user queries and filters.

The module uses TF-IDF vectorization and cosine similarity for generating recommendations.

Classes:
    ProductRecommender: Main class for product recommendation functionality.

Functions:
    load_data: Load preprocessed data from a CSV file.
    preprocess_text: Preprocess text data for analysis.
    save_search_history: Save search history to a CSV file.

Constants:
    CSV_FILE_PATH: Path to the search history CSV file.
    PREPROCESSED_DATA_PATH: Path to the preprocessed data ZIP file.

Dependencies:
    - pandas
    - nltk
    - scikit-learn
    - streamlit

Author: [Ziba]
Date: [Atak]
Version: 2.0
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import csv
import logging
from typing import List, Tuple, Optional
import os



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Constants
CSV_FILE_PATH = "search_history.csv"
PREPROCESSED_DATA_PATH = "preprocessed_data.zip"

@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load preprocessed data from a ZIP file.

    Returns:
        pd.DataFrame: Loaded DataFrame containing grocery product data.

    Raises:
        FileNotFoundError: If the preprocessed data file is not found.
    """
    try:
        return pd.read_csv(PREPROCESSED_DATA_PATH)
    except FileNotFoundError:
        logger.error(f"Preprocessed data file not found: {PREPROCESSED_DATA_PATH}")
        st.error("Data file not found. Please check the file path.")
        return pd.DataFrame()

def preprocess_text(text: str) -> str:
    """
    Preprocess the input text by removing special characters, converting to lowercase,
    tokenizing, and removing stopwords.

    Args:
        text (str): Input text to preprocess.

    Returns:
        str: Preprocessed text.
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return " ".join(filtered_tokens)

class ProductRecommender:
    """
    A class for recommending grocery products based on user queries and filters.

    Attributes:
        df (pd.DataFrame): DataFrame containing grocery product data.
        tfidf_vectorizer (TfidfVectorizer): TF-IDF vectorizer for text analysis.
        text_embeddings (scipy.sparse.csr_matrix): TF-IDF embeddings of product descriptions.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the ProductRecommender with a DataFrame of grocery products.

        Args:
            df (pd.DataFrame): DataFrame containing grocery product data.
        """
        self.df = df
        self.tfidf_vectorizer = TfidfVectorizer()
        self.text_embeddings = self.tfidf_vectorizer.fit_transform(df['consolidated_text'])

    def search_products(self, query: str, top_n: int = 5, store: Optional[str] = None, 
                        pricing: Optional[str] = None, nutritional_tags: Optional[str] = None) -> pd.DataFrame:
        """
        Search for products based on a query and optional filters.

        Args:
            query (str): Search query string.
            top_n (int, optional): Number of top recommendations to return. Defaults to 5.
            store (str, optional): Filter by store name. Defaults to None.
            pricing (str, optional): Filter by pricing category. Defaults to None.
            nutritional_tags (str, optional): Filter by nutritional tags. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing recommended products.
        """
        query = preprocess_text(query)
        query_embedding = self.tfidf_vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_embedding, self.text_embeddings)
        top_indices = similarity_scores.argsort()[0][-top_n:][::-1]
        recommendations = self.df.iloc[top_indices]

        if store:
            recommendations = recommendations[recommendations['STORE_NAME'] == store]
        if pricing:
            recommendations = recommendations[recommendations['PRICE_TAGS'] == pricing]
        if nutritional_tags:
            recommendations = recommendations[recommendations['nutritional_tags'].str.contains(nutritional_tags, na=False)]

        return recommendations

def save_search_history(search_history: List[Tuple[str, pd.DataFrame]]):
    """
    Save the search history to a CSV file.

    Args:
        search_history (List[Tuple[str, pd.DataFrame]]): List of tuples containing search queries and their corresponding recommendations.

    Raises:
        IOError: If there's an error writing to the CSV file.
    """
    try:
        with open(CSV_FILE_PATH, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Search Query", "Product Name", "Price", "Brand"])
            for query, recommendations in search_history:
                for _, row in recommendations.iterrows():
                    writer.writerow([query, row['PRODUCT_NAME_T'], row['PRODUCT_PRICE'], row['PRODUCT_BRAND']])
        logger.info(f"Search history saved to {CSV_FILE_PATH}")
    except IOError as e:
        logger.error(f"Error saving search history: {e}")
        st.error("Failed to save search history. Please try again.")