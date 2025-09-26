# IMPORT
import pandas as pd
import pickle
import streamlit as st
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.probability import FreqDist

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# OBJECTS
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
engstpwrds = set(stopwords.words('english'))

# FUNCTIONS
def preprocess_text(sentence):
  wordlist = word_tokenize(sentence)
  wordlist = [word.lower() for word in wordlist]
  nosw = [word for word in wordlist if word not in engstpwrds]
  nopuct = [token for token in nosw if token.isalpha()]
  stemmed = [stemmer.stem(token) for token in nopuct]
  lemmatized = [lemmatizer.lemmatize(token) for token in stemmed]
  return lemmatized

@st.cache_data
def loaddata():
   data = pd.read_csv('imdb-movies-dataset.csv')
   data = data.dropna()
   data['Sentiment'] = data['Rating'].apply(lambda x: 'Positive' if x>5 else 'Negative')
   return data

dataset = loaddata()

@st.cache_resource
def loadmodel():
    with open('model.pickle', 'rb') as f:
      classifier = pickle.load(f)
    return classifier

classifier = loadmodel()

allreview = " ".join(dataset['Review'])
alltokens = word_tokenize(allreview)
freqdist = FreqDist(alltokens)

def extractfeature(review):
   features = {}
   for word in freqdist.keys():
      features[word] = (word in review)
   return features

def tfidf(query, n=5):
  vectorizer = TfidfVectorizer(stop_words='english')
  tfidfmatrix = vectorizer.fit_transform(dataset['Review'])
  query_vec = vectorizer.transform([query])
  similarity = cosine_similarity(query_vec, tfidfmatrix).flatten()
  dataset['Similarity'] = similarity
  dataset_sorted = dataset.sort_values(by='Similarity', ascending=False)
  return dataset_sorted.head(n)[['Title']]


# UI
st.title("Movie Review Analysis and Recommendation System")
st.write("This application analyzes movie reviews and provides recommendations based on user input.")

revinput = st.text_area("Enter your movie review here (≥20 words):")

if st.button("Analyze Review"):
    if len(word_tokenize(revinput)) < 20:
       st.warning("Please enter a review with at least 20 words.")
    else:
       processed = preprocess_text(revinput)
       features = extractfeature(processed)
       sentiment = classifier.classify(features)
       st.subheader("Sentiment Analysis: ")
       st.write(f"Your Review: {sentiment}")

       st.subheader("Movie Recommendation:")
       recmovie = tfidf(revinput, n=5)
       for idx, row in recmovie.iterrows():
          st.write(f"{row['Title']}")
       
    #    st.subheader("Named Entity Recognition:")
    #    entities = extract_entities(revinput)
    #    if entities:
    #       for label, ents in entities.items():
    #         st.write(f"{label}: {', '.join(set(ents))}")
    #    else:
    #       st.info("No named entities found in the review.")