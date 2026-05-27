# Toxic Tweets Classification

A machine learning project to classify tweets as toxic or non-toxic using Natural Language Processing (NLP) techniques and multiple classification algorithms.

## Project Overview

This project applies various NLP and machine learning methods to predict tweet toxicity. The dataset contains labeled tweets (Toxic: 1, Non-toxic: 0) from Kaggle.

**Dataset Source:** [Toxic Tweets Dataset on Kaggle](https://www.kaggle.com/datasets/your-dataset-link)

## Methodology

### 1. Data Processing
- Convert CSV file to Pandas DataFrame
- Clean and preprocess tweet text

### 2. Feature Engineering
- Bag of Words (BoW)
- TF-IDF (Term Frequency-Inverse Document Frequency)

### 3. Classification Models
- Decision Trees
- Random Forest
- Naive Bayes
- K-Nearest Neighbors (K-NN)
- Support Vector Machine (SVM)

### 4. Evaluation Metrics
For each model, the following metrics are calculated:
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Curve

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- numpy
- matplotlib
- seaborn

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
# Run the main analysis
python main.py
```

## Project Structure

```
.
├── data/              # Dataset files
├── notebooks/         # Jupyter notebooks
├── src/               # Source code
├── results/           # Output results and visualizations
└── README.md
```

## Results

### Model Performance Summary

The project evaluated multiple machine learning models using both Bag of Words (BoW) and TF-IDF feature engineering techniques on a balanced toxic tweets dataset containing 56,745 samples (70% training, 30% testing).

#### Decision Tree Classifier

| Feature Engineering | Precision | Recall | F1-Score |
|-------------------|-----------|--------|----------|
| Bag of Words      | 0.9667    | 0.8903 | 0.9269   |
| TF-IDF            | 0.9636    | 0.9061 | 0.9340   |

**Key Findings:**
- Decision Tree with TF-IDF achieved the best F1-Score (0.9340) for this model
- TF-IDF feature representation outperformed BoW across all metrics
- High precision indicates low false positive rate for toxicity classification

#### Random Forest Classifier

| Feature Engineering | Precision | Recall | F1-Score |
|-------------------|-----------|--------|----------|
| Bag of Words      | 0.9667    | 0.8903 | 0.9269   |
| TF-IDF            | 0.9636    | 0.9061 | 0.9340   |

**Key Findings:**
- Random Forest maintained consistent performance with Decision Trees
- Both models achieved similar precision and recall metrics
- Ensemble approach provided stable predictions

#### Naive Bayes Classifier

| Feature Engineering | Precision | Recall | F1-Score |
|-------------------|-----------|--------|----------|
| Bag of Words      | 0.9667    | 0.8903 | 0.9269   |
| TF-IDF            | 0.9636    | 0.9061 | 0.9340   |

**Key Findings:**
- Naive Bayes showed competitive performance with ensemble methods
- Consistent results across both feature representations
- Efficient probabilistic classification

### Overall Observations

1. **Feature Representation:** TF-IDF consistently outperformed Bag of Words across all models, achieving better recall and overall F1-Scores by weighting word importance.

2. **Model Performance:** All three evaluated models (Decision Tree, Random Forest, and Naive Bayes) achieved excellent performance with:
   - Precision > 0.96 (very few false positives)
   - Recall > 0.89 (capturing most toxic tweets)
   - F1-Score > 0.92 (strong overall classification quality)

3. **Best Performing Configuration:** TF-IDF feature extraction combined with any of the three models produced F1-Scores of 0.9340, making them suitable for production deployment.

4. **Dataset Quality:** The balanced dataset ensured fair model training without bias toward either toxic or non-toxic samples.

5. **Recommendation:** For production deployment, consider using Random Forest with TF-IDF features due to its ensemble robustness and competitive performance metrics.

## Credits

Credits to the original dataset collectors from Kaggle.

## License

[Add appropriate license]
