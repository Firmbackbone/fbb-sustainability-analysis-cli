import json
import pandas as pd
from keyword_extractor import KeywordExtractor
from index_generator import IndexGenerator

# Read test data
test_data = []
with open('data/test.ndjson', 'r') as f:
    for line in f:
        try:
            test_data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error decoding line: {e}")
            continue

# Convert to DataFrame
df_text = pd.DataFrame(test_data)

# Test KeywordExtractor
print("\nTesting KeywordExtractor...")
try:
    keyword_extractor = KeywordExtractor(n_keywords=10)
    df_metrics = pd.read_csv('data/scores.csv')
    result = keyword_extractor.run(df_text, df_metrics, metric_col='final_score')
    print("\nKeyword extraction results:")
    print(result.head())
except Exception as e:
    print(f"Error in KeywordExtractor: {e}")

# Test IndexGenerator
print("\nTesting IndexGenerator...")
try:
    index_generator = IndexGenerator(w_simple=0.4, w_advanced=0.4, w_sent=0.2)
    keywords = ['sustainability', 'environment', 'green', 'eco-friendly']  # Example keywords
    result = index_generator.run(df_text, keywords)
    print("\nIndex generation results:")
    print(result.head())
except Exception as e:
    print(f"Error in IndexGenerator: {e}") 