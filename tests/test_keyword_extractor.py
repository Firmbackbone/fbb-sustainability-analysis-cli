import json

import pandas as pd

from index_generator import IndexGenerator
from keyword_extractor import KeywordExtractor

# Read test data with better error handling
test_data = []
with open("data/test.ndjson", "r") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        try:
            test_data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error decoding line {i + 1}: {e}")
            continue

# Convert to DataFrame
df_text = pd.DataFrame(test_data)
if df_text.empty:
    print("Warning: No valid data was loaded from test.ndjson")

# Test KeywordExtractor
print("\nTesting KeywordExtractor...")
try:
    # Make sure the metrics file exists
    try:
        df_metrics = pd.read_csv("data/scores.csv")
    except Exception as e:
        print(f"Error reading metrics file: {e}")
        print("Running index_generator first to create scores.csv...")

        # Generate scores if not available
        index_generator = IndexGenerator(w_simple=0.4, w_advanced=0.4, w_sent=0.2)
        keywords = ["sustainability", "environment", "green", "eco-friendly"]
        result = index_generator.run(df_text, keywords)
        df_metrics = result
        df_metrics.to_csv("data/scores.csv", index=False)

    keyword_extractor = KeywordExtractor(n_keywords=10)
    result = keyword_extractor.run(df_text, df_metrics, metric_col="final_score")
    print("\nKeyword extraction results:")
    print(result.head())
except Exception as e:
    print(f"Error in KeywordExtractor: {e}")

# Test IndexGenerator
print("\nTesting IndexGenerator...")
try:
    index_generator = IndexGenerator(w_simple=0.4, w_advanced=0.4, w_sent=0.2)
    keywords = [
        "sustainability",
        "environment",
        "green",
        "eco-friendly",
    ]  # Example keywords
    result = index_generator.run(df_text, keywords)
    print("\nIndex generation results:")
    print(result.head())
except Exception as e:
    print(f"Error in IndexGenerator: {e}")
