import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from IPython.display import Image, display, HTML
import csv


search_history = []


# Load the preprocessed DataFrame
@st.cache_data
def load_data():
    return pd.read_csv("preprocessed_data.zip")
selected_df = load_data()

# Preprocessing function
def preprocess_text(text):
    # Remove special characters and convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into a string
    processed_text = " ".join(filtered_tokens)
    return processed_text

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit the vectorizer on the consolidated textual data
text_embeddings = tfidf_vectorizer.fit_transform(selected_df['consolidated_text'])

def search_products(query, top_n=5, store=None, pricing=None, nutritional_tags=None):
    # Preprocess the query
    query = preprocess_text(query)
    # Transform the query into an embedding using the TF-IDF vectorizer
    query_embedding = tfidf_vectorizer.transform([query])
    # Calculate the cosine similarity between the query embedding and all product embeddings
    similarity_scores = cosine_similarity(query_embedding, text_embeddings)
    # Get the indices of the top-N most similar products
    top_indices = similarity_scores.argsort()[0][-top_n:][::-1]
    # Retrieve the top-N recommended products
    recommendations = selected_df.iloc[top_indices]
    
    # Filter recommendations based on store
    if store:
        recommendations = recommendations[recommendations['STORE_NAME'] == store]
    
    # Filter recommendations based on pricing label
    if pricing:
        recommendations = recommendations[recommendations['PRICE_TAGS'] == pricing]
    
    # Filter recommendations based on nutritional tags
    if nutritional_tags:
        recommendations = recommendations[recommendations['nutritional_tags'].str.contains(nutritional_tags)]
    
    # Check if there are any recommendations
    if len(recommendations) == 0:
        st.warning("No products matching the filters.")
        return
    
    # Store the search query and recommendations in the search history
    search_history.append((query, recommendations))
    
    # Generate HTML representation of the results
    html_output = "<table>"
    
    for index, row in recommendations.iterrows():
        html_output += "<tr>"
        html_output += f"<td><a href='{row['PRODUCT_LINK']}' target='_blank'><img src='{row['IMAGE_URL']}' style='width:150px;height:150px;'></a></td>"
        html_output += "<td>"
        html_output += f"<b>Product Name:</b> {row['PRODUCT_NAME_T']}<br>"
        html_output += f"<b>Price:</b> {row['PRODUCT_PRICE']}<br>"
        html_output += f"<b>Brand:</b> {row['PRODUCT_BRAND']}<br>"
        
        # Retrieve the stores that carry the product
        stores = selected_df[selected_df['PRODUCT_NAME_T'] == row['PRODUCT_NAME_T']]['STORE_NAME'].tolist()
        html_output += f"<b>Stores:</b> {', '.join(stores)}<br>"
        
        html_output += f"<b>Pricing Category:</b> {row['PRICE_TAGS']}<br>"
        html_output += "</td>"
        html_output += "</tr>"
    
    html_output += "</table>"
    
    # Display the HTML output
    st.markdown(html_output, unsafe_allow_html=True)

# Custom CSS styling
st.markdown(
    """
    <style>
    /* Title font */
    .title {
        font-family: 'Montserrat Bold', sans-serif;
        font-size: 25px;
    }
    
    /* Text font */
    body {
        font-family: 'Montserrat Classic', sans-serif;
        font-size: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create the search form using Streamlit
st.title("Grocery Recommendation App")

search_query = st.text_input("Search:")
top_n = st.number_input("Number of recommendations:", min_value=1, max_value=30, value=5)
store = st.selectbox("By Store (optional):",[None, 'Wolt: Flink Karl Liebknecht', 'Muller', 'EDEKA', 'Amazon', 'REWE',
       'Flink', 'Mitte Meer Charlottenburg', 'Asia Food Tuan Lan',
       'Wolt: Latino Point', 'Amore Store', 'Asia24 GmbH',
       'Goldhahn & sampson', 'Oda', 'Wolt: Miconbini', "Golly's", 'PENNY',
       'FLINK', 'Tante Emma', 'Mitte Meer', 'ASIA4FRIENDS', 'Veganz',
       'Original Unverpackt'])
pricing = st.selectbox("By Price Level (optional):", [None, "Budget", "Premium", "Mid-range"])
nutritional_tags = st.selectbox("By nutritional tags (optional):",[None,'Source of Protein', 'High Protein', 'Low Sugars', 'Low Sodium', 'Low Fat', 'Fat Free'])

if st.button("Search"):
    search_products(query=search_query, top_n=top_n, store=store, pricing=pricing, nutritional_tags=nutritional_tags)
   
# Specify the file path for saving the search history
csv_file_path = "search_history.csv"

# Save the search history to a CSV file
with open(csv_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Search Query", "Product Name", "Price", "Brand"])  # Write the header
    for query, recommendations in search_history:
        for index, row in recommendations.iterrows():
            writer.writerow([query, row['PRODUCT_NAME_T'], row['PRODUCT_PRICE'], row['PRODUCT_BRAND']])

