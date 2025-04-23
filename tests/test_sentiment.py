import json
import pandas as pd
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

# Test keywords (sustainability-related)
keywords = [
    'sustainability', 'environment', 'green', 'eco-friendly',
    'renewable', 'energy', 'climate', 'carbon', 'emission',
    'recycling', 'waste', 'conservation', 'biodiversity'
]

# Initialize IndexGenerator with equal weights
index_generator = IndexGenerator(w_simple=0.4, w_advanced=0.4, w_sent=0.2)

# Run the test
print("\nTesting IndexGenerator with new sentiment analysis...")
print(f"Processing {len(df_text)} documents...")
results = index_generator.run(df_text, keywords)

# Print summary statistics
print("\nSummary Statistics:")
print(f"Total documents processed: {len(results)}")
print(f"Documents with matches: {len(results[results['simple_match_score'] > 0])}")
print(f"Average sentiment score: {results['sentiment_score'].mean():.4f}")
print(f"Average final score: {results['final_score'].mean():.4f}")

# Print top 5 results
print("\nTop 5 Results:")
top_results = results.sort_values('final_score', ascending=False).head()
for _, row in top_results.iterrows():
    print(f"\nDomain: {row['domain']}")
    print(f"Simple Match Score: {row['simple_match_score']}")
    print(f"Advanced Match Score: {row['advanced_match_score']}")
    print(f"Sentiment Score: {row['sentiment_score']}")
    print(f"Final Score: {row['final_score']}")
    print("-" * 80)

# Print bottom 5 results
print("\nBottom 5 Results:")
bottom_results = results.sort_values('final_score').head()
for _, row in bottom_results.iterrows():
    print(f"\nDomain: {row['domain']}")
    print(f"Simple Match Score: {row['simple_match_score']}")
    print(f"Advanced Match Score: {row['advanced_match_score']}")
    print(f"Sentiment Score: {row['sentiment_score']}")
    print(f"Final Score: {row['final_score']}")
    print("-" * 80) 