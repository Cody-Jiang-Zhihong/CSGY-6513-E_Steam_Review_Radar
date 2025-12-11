# Steam Review Radar

## Introduction
Dataset: [100 Million+ Steam Reviews (Kaggle)](https://www.kaggle.com/datasets/kieranpoc/steam-reviews/data)
Size: 113,883,717 reviews (42.49GB)
Names:
- Andrew Wang
- Cody Jiang
- Dylan Zhou

In this project we take a slice of the dataset and analyze the Steam game reviews to predict review helpfulness and identify reviewer archetypes. The analysis includes spam detection, helpfulness prediction using machine learning, and unsupervised clustering to discover reviewer personas.

### Project Structure
```
├── 01_preprocessing.ipynb
├── 02_visualization.ipynb
├── 03_helpfulness_and_reviewer_archetypes.ipynb
├── 04_spam_detection.ipynb
├── requirements.txt
└── steam_dataset
    ├── sml_sample.csv
    └── sml_sample.parquet
```

## Overview

### Prerequisites
All the prerequisites for running the pipeline are in the `requirements.txt`. It is recommended to create a Python virtual environment before installing.
```
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

System Requirements:
- RAM: 8GB minimum
- storage: ~80GB to store the dataset versions


### Data Preprocessing (`01_preprocessing.ipynb`)
Downsamples and filters the original dataset to create manageable subsets.
- Downsample to 10%: reduces full dataset to ~10% using random sampling
- Filter: extracts English-only reviews and drops China-specific columns
- Create medium/small samples:
    - Medium: ~2M reviews
    - Small: ~20K reviews (subset of medium)
- Convert to Parquet: converts CSV to Parquet format for efficient storage


### Data Visualization (`visualization.ipynb`)
Visualize parts of dataset to get an understanding of its characteristics.
- Weighted vote score distribution
- User playtime histogram (log scale)
- Number of games owned distribution
- Purchase behavior (Steam purchase vs free keys)
- Early access review distribution
- Top 50 most-reviewed games

### Helpfulness Prediction and Reviewer Archetypes (`03_helpfulness_and_reviewer_archetypes.ipynb`)
Predicts the review helpfulness and discovers reviewer personas.

#### Part A: Helpfulness Prediction
Model Architecture:
- Preprocessing Pipeline:
    - TF-IDF on review text (3000 features, unigrams + bigrams)
    - StandardScaler for numeric features
    - OneHotEncoder for categorical features
- Model: Poisson Regression (appropriate for count data)
- Features:
    - Text: Review content
    - Numeric: Games owned, playtime, votes\_funny, weighted\_vote\_score, etc.
    - Categorical: Language, purchase type, early access status

Evaluation Visualizations:
- True vs Predicted scatter plot
- Residual analysis
- Distribution comparison
- Calibration plot (5-bin)

#### Part B: Reviewer Archetypes
- Feature Engineering: Aggregate reviews by author to create reviewer-level features:
    - Review volume and game ownership
    - Average playtime (total and recent)
    - Average helpfulness and funny votes
    - Purchase behavior ratios
    - Recommendation positivity
- MiniBatchKMeans Clustering:
    - 5 clusters identified
    - Features standardized before clustering
    - Validated with PCA 2D projection

Personas:
| Cluster | Size | Key Characteristics | Persona Label |
|---------|------|---------------------|---------------|
| 0 | 910,219 (56%) | Moderate library (85 games), regular reviewers (11 reviews), deep playtime (11K min), mostly purchased games (97%), very positive (89% thumbs up), low helpfulness (1.5 votes avg) | Casual Enthusiasts |
| 1 | 63,172 (4%) | Moderate library (124 games), regular reviewers (13 reviews), deep playtime (14K min), heavy free-key usage (92%), very positive (90% thumbs up), low helpfulness (1.8 votes avg) | Free-Key Reviewers |
| 2 | 418,583 (26%) | Moderate library (112 games), regular reviewers (11 reviews), deep playtime (13K min), mixed purchase behavior, very positive (88% thumbs up), low helpfulness (1.5 votes avg) | Balanced Gamers |
| 3 | 34,449 (2%) | Small library (65 games), rarely writes reviews (5 reviews), EXTREME playtime (231K min = 3,855 hours), mixed purchase behavior, very positive (90% thumbs up), moderately high helpfulness (3.0 votes avg) | Core Players |
| 4 | 199,100 (12%) | Large library (127 games), most active reviewers (15 reviews), deep playtime (10K min), mixed purchase behavior, critical/negative (only 5% thumbs up!), HIGHEST helpfulness (6.97 votes avg) | Critical Veterans |

### Spam Detection (`spam_detection.ipynb`)
Identifies near-duplicate reviews and style anomalies using MinHash LSH and Isolation Forest.
- Text Cleaning and Shingle (n-grams) Creation:
    - Lowercase and remove special characters
    - Generate character 5-grams for similarity detection
- MinHash LSH for Near-Duplicates:
    - Build MinHash signatures (128 permutations)
    - Create LSH index with 0.8 similarity threshold
    - Identify 87,267 near-duplicate pairs (~14.7% of reviews)
- Style Anomaly Detection:
    - Extract style features (word count, unique ratio, avg word length)
    - Use Isolation Forest to detect unusual writing patterns
    - Flag ~2% of reviews as style anomalies
- Spam Flag Integration:
    - Add near\_duplicate\_flag and style\_anomaly\_flag columns

Compare three approaches:
- Baseline: No spam handling
- Spam flags as features: Include flags in model
- Down-weighting: Reduce weight of spam samples during training

Results:
- Baseline RMSE: 6.373
- With spam flags: 6.231 RMSE (-2.2% improvement)
- With down-weighting: 6.218 RMSE (-2.4% improvement)
- Spearman correlation improves from 0.694 to 0.695

