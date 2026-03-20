from typing import Any


from dotenv import load_dotenv
import openai
import streamlit as st
import os
import pandas as pd
import re

# @st.cache_data
# def get_response(user_prompt, temperature):
#     response = client.responses.create(
#         model="gpt-4o",
#         input=[
#             {"role": "user", "content":user_prompt}
#         ],
#         temperature=temperature, # Change the probability for creativity
#         max_output_tokens=100 # limit response length
#     )
#     return response


def clean_text(text: str) -> str:
    # Lowercase and trim leading/trailing whitespace
    text = text.lower().strip()
    # Remove punctuation (keeps letters, numbers, and whitespace)
    text = re.sub(r"[^\w\s]", "", text)
    return text

def get_dataset_path():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the csv file
    csv_path = os.path.join(current_dir, "data", "customer_reviews.csv")
    return csv_path  

# load environment variables from .env file
load_dotenv()


st.title("Hello, GenAI")
st.write("Visualize your data with GenAI.")

# Layout two buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button(" 📥 Ingest Dataset"):
        try: 
            csv_path = get_dataset_path()
            st.session_state['df'] = pd.read_csv(csv_path)
            st.success("Dataset ingested successfully")
        except FileNotFoundError:
            st.error(f"Dataset not found. Please check the file path.")
with col2:
    if st.button(" 📄 Parse Reviews"):
        if 'df' in st.session_state:
            st.session_state['df']['Cleaned_Summary'] = st.session_state['df']['SUMMARY'].apply(clean_text)
            st.success("Reviews parsed successfully")
        else:
            st.warning("Please ingest the dataset first.")
# Display the dataset if it exists
if 'df' in st.session_state:
    st.subheader(" 🔍 Filter by Product")
    product_filter = st.selectbox("Select a product", ["All Products"] + st.session_state['df']['PRODUCT'].unique().tolist())
    st.subheader(" 📁 Dataset Preview")

    if product_filter != "All Products":
        filtered_df = st.session_state['df'][st.session_state['df']['PRODUCT'] == product_filter]
        st.success(f"Filtered by {product_filter}")
    else:
        filtered_df = st.session_state['df']
    st.dataframe(filtered_df)

    st.subheader(" 📊 Sentiment Score by Product")
    grouped = st.session_state['df'].groupby('PRODUCT')['SENTIMENT_SCORE'].mean().reset_index()
    st.bar_chart(grouped, x='PRODUCT', y='SENTIMENT_SCORE')


