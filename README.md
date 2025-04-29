![FBB_Logo](https://firmbackbone.nl/wp-content/uploads/sites/694/2025/03/FBB-logo-wide.png)

**FIRMBACKBONE** is an organically growing longitudinal data-infrastructure with information on Dutch companies for scientific research and education. Once it is ready, it will become available for researchers and students affiliated with Dutch member universities through the Open Data Infrastructure for Social Science and Economic Innovations ([ODISSEI](https://odissei-data.nl/nl/)). FIRMBACKBONE is an initiative of Utrecht University ([UU](https://www.uu.nl/en)) and the Vrije Universiteit Amsterdam ([VU Amsterdam](https://vu.nl/en)) funded by the Platform Digital Infrastructure-Social Sciences and Humanities ([PDI-SSH](https://pdi-ssh.nl/en/front-page/)) for the period 2020-2025.

# fbb-sustainability-analysis-cli
FBB Sustainability Analysis command line interface

This package provides two main tools for analyzing sustainability-related content in websites:

1. **Keyword Extractor**: Identifies important sustainability-related keywords from text data
2. **Index Generator**: Scores websites based on their sustainability content

## Quick Start

### 1. Keyword Extractor

The Keyword Extractor helps you find important sustainability-related terms in your text data.

```bash
python keyword_extractor.py --text-data data/pages.ndjson \
                           --metrics data/metrics.csv \
                           --output data/keyword_imp.csv \
                           --n-keywords 25
```

Options:
- `--text-data`: Your input file containing website text (NDJSON format)
- `--metrics`: File containing metrics/scores for the websites
- `--output`: Where to save the extracted keywords
- `--n-keywords`: How many top keywords to extract (default: 20, max: 50)
- `--test-size`: Portion of data to use for testing (default: 0.2 or 20%)

Example output:
```
keyword,importance
sustainability,0.85
renewable,0.72
green energy,0.68
...
```

### 2. Index Generator

The Index Generator scores websites based on their sustainability content using three factors:
- Simple keyword matches
- Advanced matches (including synonyms)
- Sentiment analysis of sustainability-related content

```bash
python index_generator.py --data data/pages.ndjson \
                         --keywords data/keywords.csv \
                         --output data/scores.csv \
                         --simple-weight 0.4 \
                         --advanced-weight 0.4 \
                         --sentiment-weight 0.2
```

Options:
- `--data`: Your input file containing website text (NDJSON format)
- `--keywords`: File containing sustainability keywords to look for
- `--output`: Where to save the scoring results
- `--simple-weight`: Weight for direct keyword matches (default: 0.3)
- `--advanced-weight`: Weight for synonym matches (default: 0.4)
- `--sentiment-weight`: Weight for sentiment analysis (default: 0.3)

To calculate the final score, weights assigned to the simple, advanced and sentiment scores are used. The final score is calculated by multiplying the given weight with the retrieve score summed up per analysis type.

Example output:
```
domain,simple_match_score,advanced_match_score,sentiment_score,final_score
https://example.com,3,3.5,0.8,2.9
...
```

## Understanding the Scores

### Keyword Extractor
- Extracts the most important sustainability-related terms from your text
- Uses machine learning to identify significant keywords
- Filters out common words and names
- Supports multiple languages (English and Dutch)

### Index Generator
The final score is calculated using three components:

1. **Simple Match Score**
   - Counts direct matches of sustainability keywords
   - Weight: 0.4 (adjustable)

2. **Advanced Match Score**
   - Includes matches of keyword synonyms
   - Each synonym match counts as 0.5
   - Weight: 0.4 (adjustable)

3. **Sentiment Score**
   - Analyzes the sentiment of sentences containing sustainability keywords
   - Only calculated when keywords are found
   - Weight: 0.2 (adjustable)

## Data Format Requirements

### Input Files
1. **Text Data (NDJSON format)**
   ```json
   {"domain": "https://example.com", "text": "Website content..."}
   ```

2. **Keywords (CSV format)**
   ```
   keyword
   sustainability
   renewable energy
   green technology
   ```

3. **Metrics (CSV format, for Keyword Extractor)**
   ```
   domain,score
   https://example.com,0.85
   ```

### Output Files
1. **Keyword Importance (CSV)**
   ```
   keyword,importance
   sustainability,0.85
   ```

2. **Website Scores (CSV)**
   ```
   domain,simple_match_score,advanced_match_score,sentiment_score,final_score
   https://example.com,3,3.5,0.8,2.9
   ```

## Tips for Best Results

1. **For Keyword Extraction**
   - Use at least 50-100 websites for good keyword extraction
   - Include both high and low scoring websites
   - Clean your text data of irrelevant content

2. **For Index Generation**
   - Use a comprehensive list of sustainability keywords
   - Adjust weights based on your priorities
   - Consider language differences (English/Dutch)

## Common Questions

Q: Why are some websites getting zero scores?
A: Websites get zero scores when:
- No sustainability keywords are found
- The content is in a language not supported for sentiment analysis
- The text doesn't contain relevant sustainability content

Q: How do I adjust the scoring?
A: Use the weight parameters:
- Increase `--simple-weight` to focus on direct keyword matches
- Increase `--advanced-weight` to value synonym matches more
- Increase `--sentiment-weight` to emphasize the tone of sustainability content

Q: What languages are supported?
A: Both tools support English and Dutch content, with special handling for:
- Language detection
- Stop word removal
- Named entity recognition
- Sentiment analysis

## Machine Learning Components

### Keyword Extractor
The Keyword Extractor uses several machine learning techniques to identify important sustainability terms:

1. **Text Preprocessing**
   - Tokenization: Breaks text into words and phrases
   - Lemmatization: Reduces words to their base form (e.g., "sustainable" → "sustain")
   - Stop Word Removal: Filters out common words
   - Named Entity Recognition: Removes person and organization names

2. **Feature Extraction**
   - TF-IDF Vectorization: Converts text into numerical features
     - max_features=6000: Limits to most important terms
     - min_df=1: Includes terms appearing at least once
     - max_df=0.95: Excludes very common terms

3. **Model Selection**
   The tool automatically chooses between two models based on your data:
   - **Regression Model** (RandomForestRegressor)
     - Used when analyzing continuous scores
     - 400 decision trees
     - Parallel processing for speed
   
   - **Classification Model** (RandomForestClassifier)
     - Used when analyzing categorical scores
     - 400 decision trees
     - Handles multiple categories

4. **Feature Importance**
   - Uses built-in feature importance from Random Forest
   - Ranks terms by their contribution to predictions
   - Filters out entity names and common words
   - Returns top N most important keywords

### Index Generator
The Index Generator uses natural language processing (NLP) techniques:

1. **Text Processing**
   - Language Detection: Identifies English or Dutch content
   - Tokenization: Splits text into words
   - Lemmatization: Standardizes word forms
   - Sentence Splitting: Identifies sentence boundaries

2. **Keyword Matching**
   - Simple Matching: Direct word matches
   - Advanced Matching: Uses WordNet for synonyms
   - Example: "sustainable" matches with "eco-friendly", "green"

3. **Sentiment Analysis**
   - Uses VADER (Valence Aware Dictionary and sEntiment Reasoner)
   - Analyzes sentiment of sentences containing keywords
   - Provides compound sentiment scores 
   - Only analyzes relevant portions of text

### Model Performance

1. **Keyword Extractor**
   - Training/Testing Split: 80/20 by default
   - Performance Metrics:
     - Regression: R² score (0-1, higher is better)
     - Classification: Precision, Recall, F1-score
   - Example Output:
     ```
     R² = 0.85  # For regression
     precision    recall  f1-score   support  # For classification
         0.85      0.82      0.83       100
     ```

2. **Index Generator**
   - No training required
   - Real-time processing
   - Language-specific handling:

### Tips for Better ML Results

1. **Data Quality**
   - Ensure clean, relevant text content
   - Include diverse examples
   - Balance positive and negative cases

2. **Keyword Extraction**
   - Use more data for better keyword identification
   - Include both high and low scoring examples
   - Consider domain-specific terms

3. **Index Generation**
   - Use comprehensive keyword lists
   - Consider industry-specific terminology
   - Adjust weights based on your needs 

## Offline Installation

For environments without internet access, follow these steps:

1. **On a machine with internet access:**
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd sustainability-analysis

   # Run the setup script
   chmod +x setup_offline.sh
   ./setup_offline.sh
   ```

2. **Transfer the following to the offline machine:**
   - The entire repository
   - The `environment.yml` file
   - The following directories (from your conda environment):
     - `nltk_data/`
     - `spacy/` (models directory)

3. **On the offline machine:**
   ```bash
   # Create the environment from the yml file
   conda env create -f environment.yml

   # Activate the environment
   conda activate sustainability_env

   # Copy the NLTK data
   cp -r /path/to/nltk_data ~/nltk_data

   # Copy the spaCy models
   cp -r /path/to/spacy/models/* ~/anaconda3/envs/sustainability_env/lib/python3.8/site-packages/spacy/data/
   ```

### Required Files for Offline Use

1. **NLTK Data:**
   - punkt
   - wordnet
   - vader_lexicon
   - omw-1.4

2. **spaCy Models:**
   - en_core_web_trf
   - nl_core_news_sm

### Verifying Installation

To verify the installation works offline:
```bash
# Test NLTK
python -c "import nltk; nltk.data.find('tokenizers/punkt')"

# Test spaCy
python -c "import spacy; nlp = spacy.load('en_core_web_trf')"

# Test the tools
python keyword_extractor.py --text-data data/sample_data.ndjson \
                           --metrics data/sample_metrics.csv \
                           --output data/keyword_imp.csv
``` 
