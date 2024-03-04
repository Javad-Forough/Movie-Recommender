from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text_data(data):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    text_features = tfidf_vectorizer.fit_transform(data['overview'])
    return text_features