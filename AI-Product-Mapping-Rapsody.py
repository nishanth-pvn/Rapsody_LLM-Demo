import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Title of the Streamlit app
st.set_page_config(page_title="", layout="wide")
st.markdown("<h5 style='text-align: center;'><b>Rapsody - AI Product Mapping</b></h5>", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.write(' ')
    st.sidebar.image('BI-Logo.png', width=120)
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.markdown("<h3 style='text-align: left;'><b>Distributor Product Name</b></h3>", unsafe_allow_html=True)
    new_partner_product_name = st.sidebar.text_input('Type in few letters, press enter')

# Load data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

file_path = 'Product_Mapping.csv'
data = load_data(file_path)

# Display the dataframe
#st.write('DataFrame', data.head())

# Select relevant columns
source_column = 'Partner Product Name'
target_column = 'BI Product Name'
X = data[source_column]
y = data[target_column]

# Check if model and vectorizer are already saved, if not train and save them
model_path = 'logistic_regression_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except FileNotFoundError:
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=60)

    # Text Vectorization
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train the model
    model = LogisticRegression(C=1000, max_iter=100)
    model.fit(X_train_vec, y_train)

    # Save the model and vectorizer
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

# Make predictions and evaluate the model if new data is provided
if new_partner_product_name:
    new_product = [new_partner_product_name]
    new_product_vec = vectorizer.transform(new_product)
    predicted_mapped_name = model.predict(new_product_vec)
    predicted_proba = model.predict_proba(new_product_vec)
    predicted_proba = predicted_proba / predicted_proba.sum()
    predicted_probs_percentage = {name: prob * 100 for name, prob in zip(model.classes_, predicted_proba[0])}

    # Predicted BI Product Name
    predicted_output = f"{predicted_mapped_name[0]}"
    probability_percent = dict(sorted(predicted_probs_percentage.items(), key=lambda item: item[1], reverse=True)[:1])

    # Get top three predicted BI Product Name
    top_3_predicted_probs = dict(sorted(predicted_probs_percentage.items(), key=lambda item: item[1], reverse=True)[:3])
    
    # Check the difference between the first and second record
    top_3_values = list(top_3_predicted_probs.values())
    if len(top_3_values) > 1 and (top_3_values[0] - top_3_values[1]) > 90:
        top_3_predicted_probs = {list(top_3_predicted_probs.keys())[0]: top_3_values[0]}
    
    # Filter out the third product if its probability is less than 0.5%
    top_3_predicted_probs = {k: v for k, v in top_3_predicted_probs.items() if v >= 0.5}
    
    top_3_df = pd.DataFrame(list(top_3_predicted_probs.items()), columns=['BI Product Name', 'Probability Percent (%)'])
    top_3_df.index = top_3_df.index + 1
    top_3_df['Probability Percent (%)'] = round(top_3_df['Probability Percent (%)'], 2)
    
    col1, col2 = st.columns([0.60, 0.40])
    
    with col1:
        st.markdown("<h8 style='text-align: center;'><b>Top Matching BI Product Names</b></h8>", unsafe_allow_html=True)
        st.write(top_3_df)
    
    with col2:
        st.markdown("<h8 style='text-align: center;'><b>Predicted BI Product Name</b></h8>", unsafe_allow_html=True)
        st.text(' ')
        st.text(' ')
        st.code(predicted_output, language='Python')
    
    st.text(' ')
    
    # Seaborn Visualization (Bar Chart)
    plt.figure(figsize=(18,3))
    ax = sns.barplot(data=top_3_df, x='Probability Percent (%)', y='BI Product Name', palette='viridis')
    sns.set(font_scale=1)
    sns.set_theme(style="whitegrid")
    
    # Add labels to each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_width():.1f}%', (p.get_width() + 0.1, p.get_y() + p.get_height() / 2),
                    ha='left', va='center', fontsize=10, color='black')  

    plt.title('Partner Product Name (typed-in): ' + new_partner_product_name, fontweight='bold')
    plt.xlabel('Probability Percentage (%)', fontweight='bold')
    plt.ylabel('Matched BI Product Names', fontweight='bold')
    st.pyplot(plt)

# Evaluate the model accuracy if not already evaluated
if 'X_test' in locals() and 'y_test' in locals():
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
