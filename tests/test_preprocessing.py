import json

from sklearn.feature_extraction.text import TfidfVectorizer

from keyword_extractor import preprocess

# Read a sample from the test file
with open("data/test.ndjson", "r") as f:
    sample = json.loads(f.readline())

# Get the original text
original_text = sample.get("text", "")
print("Original text:")
print(original_text[:500])  # Print first 500 characters

# Preprocess the text
processed_text = preprocess(original_text)
print("\nProcessed text:")
print(processed_text)

# Test TfidfVectorizer with adjusted parameters for single document
vectorizer = TfidfVectorizer(
    max_features=6000,
    min_df=1,
    max_df=1.0,  # Changed from 0.95 to 1.0 to handle single document
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
