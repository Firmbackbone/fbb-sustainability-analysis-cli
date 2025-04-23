import json
from keyword_extractor import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer

# Read a sample from the test file
with open('data/test.ndjson', 'r') as f:
    sample = json.loads(f.readline())

# Get the original text
original_text = sample.get('text', '')
print("Original text:")
print(original_text[:500])  # Print first 500 characters

# Preprocess the text
processed_text = preprocess(original_text)
print("\nProcessed text:")
print(processed_text)

# Test TfidfVectorizer
vectorizer = TfidfVectorizer(
    max_features=6000,
    min_df=1,
    max_df=0.95
)

# Try to fit and transform the processed text
try:
    # We need to pass a list of documents
    tfidf_matrix = vectorizer.fit_transform([processed_text])
    print("\nVectorization successful!")
    print(f"Number of features: {len(vectorizer.get_feature_names_out())}")
    print("Sample features:", vectorizer.get_feature_names_out()[:10])
except Exception as e:
    print("\nVectorization error:")
    print(str(e)) 