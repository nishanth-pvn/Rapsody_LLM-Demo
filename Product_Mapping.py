import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Title of the Streamlit app
st.set_page_config(page_title="", layout="wide")
st.markdown("<h5 style='text-align: center;'><b>Products - Mapping List</b></h5>", unsafe_allow_html=True)
st.divider()
    
# Load data
def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

file_path = 'Product_Mapping.xlsx'
data = load_data(file_path)

# Display the dataframe
st.write(data)
