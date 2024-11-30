import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import nltk
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, 
                              ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier)
from xgboost import XGBClassifier
import seaborn as sns
import pickle

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = stopwords.words('english')

# App title
st.title("Spam Detection Data Analysis")

# File upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

def load_data(file):
    try:
        df = pd.read_csv(file, encoding='latin1')
        return df
    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty or invalid.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None
# Load vectorizer and model from pickle files
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.header("Dataset Overview")
        st.write(df.head())

        st.header("Data Cleaning")

        # Step 1: Column Renaming
        st.subheader("1. Rename Columns")
        st.write("Renaming columns 'v1' to 'target' and 'v2' to 'text'.")
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore', inplace=True)
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        st.write("Renamed Columns:")
        st.write(df.head())

        # Step 2: Label Encoding
        st.subheader("2. Encoding Target Column")
        st.write("Encoding 'target' column: 'ham' -> 0, 'spam' -> 1.")
        encoder = LabelEncoder()
        df['target'] = encoder.fit_transform(df['target'])
        st.write("Encoded Data:")
        st.write(df[['target', 'text']].head())

        # Step 3: Feature Engineering
        st.subheader("3. Feature Engineering")
        st.write("Adding features: number of characters, words, and sentences.")
        df['num_characters'] = df['text'].apply(len)
        df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
        df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

        st.write("Feature Engineered Data:")
        st.dataframe(df[['text', 'num_characters', 'num_words', 'num_sentences']].head())

        # Exploratory Data Analysis
        st.header("Exploratory Data Analysis")

        st.subheader("Distribution of Ham and Spam")
        colors = ['#148f77', '#b03a2e']
        fig1, ax1 = plt.subplots()
        ax1.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f%%", colors=colors)
        st.pyplot(fig1)

        st.subheader("Character Distribution")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.histplot(df[df['target'] == 0]['num_characters'], ax=ax2, color='blue', label='Ham')
        sns.histplot(df[df['target'] == 1]['num_characters'], ax=ax2, color='red', label='Spam')
        ax2.legend()
        st.pyplot(fig2)

        st.subheader("Words Distribution")
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        sns.histplot(df[df['target'] == 0]['num_words'], ax=ax3, color='blue', label='Ham')
        sns.histplot(df[df['target'] == 1]['num_words'], ax=ax3, color='red', label='Spam')
        ax3.legend()
        st.pyplot(fig3)

        st.subheader("Correlation Feature Heatmap")
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        fig4, ax4 = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap='YlGnBu', ax=ax4)
        st.pyplot(fig4)

        # Data Preprocessing and Transformation
        ps = PorterStemmer()

        def transform_text(text):
            text = text.lower()
            text = nltk.word_tokenize(text)
            text = [word for word in text if word.isalnum() and word not in stop_words]
            text = [ps.stem(word) for word in text]
            return " ".join(text)

        df['transformed_text'] = df['text'].apply(transform_text)

        # WordClouds
        st.header("WordClouds")

        st.subheader("Spam WordCloud")
        spam_wcd = WordCloud(width=500, height=500, min_font_size=10, background_color='white').generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        ax5.imshow(spam_wcd)
        ax5.axis('off')
        st.pyplot(fig5)

        st.subheader("Ham WordCloud")
        ham_wcd = WordCloud(width=500, height=500, min_font_size=10, background_color='white').generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))
        fig6, ax6 = plt.subplots(figsize=(12, 6))
        ax6.imshow(ham_wcd)
        ax6.axis('off')
        st.pyplot(fig6)

        spam_corpus = [word for msg in df[df['target'] == 1]['transformed_text'] for word in msg.split()]
        ham_corpus = [word for msg in df[df['target'] == 0]['transformed_text'] for word in msg.split()]

        # Interactive graph selection
        st.header("Top Words in Spam and Ham Messages")
        option = st.radio("Choose which graph to display:", ("Top Spam Words", "Top Ham Words"))

        if option == "Top Spam Words":
            data = pd.DataFrame(Counter(spam_corpus).most_common(30), columns=['Word', 'Frequency'])
            st.subheader("Top 30 Most Common Spam Words")
            fig, ax = plt.subplots()
            sns.barplot(x='Word', y='Frequency', data=data, ax=ax, color='#cb4335')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        elif option == "Top Ham Words":
            data = pd.DataFrame(Counter(ham_corpus).most_common(30), columns=['Word', 'Frequency'])
            st.subheader("Top 30 Most Common Ham Words")
            fig, ax = plt.subplots()
            sns.barplot(x='Word', y='Frequency', data=data, ax=ax, color='#1a5276')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        # Feature extraction
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['transformed_text']).toarray()
    y = df['target'].values

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    # Model choices
    clfs = {
        'SVC': SVC(kernel='sigmoid', gamma=1.0),
        'KNeighbors': KNeighborsClassifier(),
        'MultinomialNB': MultinomialNB(),
        'DecisionTree': DecisionTreeClassifier(max_depth=5),
        'RandomForest': RandomForestClassifier(n_estimators=50, random_state=2),
    }

    classifier_name = st.selectbox("Choose a classifier", list(clfs.keys()))

    if st.button("Train"):
        clf = clfs[classifier_name]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        st.write(f"**Accuracy:** {accuracy:.2f}")
        st.write(f"**Precision:** {precision:.2f}")
        st.write("**Confusion Matrix:**")
        st.write(confusion_matrix(y_test, y_pred))

        # Plot performance comparison
        st.subheader("Comparison of all classifiers")
        accuracy_scores = []
        precision_scores = []
        
        for name, model in clfs.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred))

        performance_df = pd.DataFrame({
            'Algorithm': list(clfs.keys()),
            'Accuracy': accuracy_scores,
            'Precision': precision_scores
        }).sort_values('Precision', ascending=False)

        st.dataframe(performance_df)

        # Visualization
        plt.figure(figsize=(10, 5))
        sns.barplot(x='Algorithm', y='value', hue='variable', 
                    data=pd.melt(performance_df, id_vars="Algorithm"))
        plt.xticks(rotation=45)
        st.pyplot(plt)

# Real-time classification
st.header("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess
    transformed_sms = transform_text(input_sms)
    
    # Vectorize
    vector_input = tfidf.transform([transformed_sms])
    
    # Predict
    result = model.predict(vector_input)[0]
    
    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
     
