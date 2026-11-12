# Customer Review Sentiment Analyzer
### Feedback Classification & Aspect-Based Feature Extraction

An intelligent NLP application that analyzes customer reviews, classifies sentiment, extracts key product and service aspects, and compares traditional machine learning, transformer-based models, and LLM prompting strategies.

## Overview

The **Customer Review Sentiment Analyzer** is a Python-based application designed to automate the analysis of customer feedback.

It classifies reviews into three sentiment categories — **Positive, Neutral, and Negative** — while identifying key aspects associated with customer opinions.

The project combines classical machine learning, fine-tuned transformer models, and prompt engineering techniques within an interactive Streamlit interface.

## Key Features

- **Sentiment Classification:** Classifies customer reviews into positive, neutral, and negative categories.
- **Aspect & Feature Extraction:** Extracts important product/service aspects using spaCy noun chunks and associates them with sentence-level sentiment.
- **Multiple ML Approaches:** Implements TF-IDF with Logistic Regression and a fine-tuned DistilBERT classifier.
- **Prompt Engineering Lab:** Experiments with zero-shot, role-based, few-shot, structured prompting, self-consistency, and prompt chaining.
- **Single Review Analysis:** Analyzes individual customer reviews and displays predictions.
- **Bulk CSV Analysis:** Processes uploaded datasets containing multiple customer reviews.
- **Model Comparison:** Evaluates different approaches using accuracy, macro-F1, weighted-F1, and aspect-level metrics.
- **Structured Output:** Uses JSON parsing and validation with an automated repair attempt for malformed responses.
- **Prompt Injection Guard:** Includes configurable guard instructions for supported prompting techniques.
- **Visual Analytics:** Displays sentiment distributions, evaluation metrics, and comparison results.

## System Architecture

The application follows a modular architecture:

1. **Input Layer:** Accepts individual reviews or CSV datasets through Streamlit.
2. **Preprocessing:** Cleans review text, handles missing values, and maps ratings to sentiment labels.
3. **Model Layer:** Routes inputs through the selected ML classifier or LLM-based workflow.
4. **Feature Extraction:** Identifies relevant aspects and associates them with sentiment.
5. **Evaluation & Visualization:** Generates predictions, calculates performance metrics, and displays comparison results.

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| NLP | spaCy |
| Deep Learning | PyTorch, Hugging Face Transformers |
| Models | Logistic Regression, DistilBERT, Qwen2.5 Instruct |
| Prompt Engineering | Zero-shot, Few-shot, CoT, Self-Consistency, Prompt Chaining |
| Interface | Streamlit |
| Visualization | Matplotlib |
| Testing | Pytest |

## Model Evaluation

The project evaluates multiple approaches using real-world Amazon customer review data and a separate hand-labelled evaluation set.

| Approach | Accuracy | Macro-F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 89.1% | 70.2% |
| Fine-tuned DistilBERT | 91.7% | 61.2% |
| Zero-shot Prompting | 83.3% | 82.2% |
| Few-shot Prompting | 91.7% | 91.5% |
| Structured Prompting | 100% | 100% |

*Note: Prompt comparison results are based on a completed 12-review evaluation using Qwen2.5-0.5B-Instruct. These preliminary results should not be interpreted as generalizable model performance.*

## Project Structure

```text
customer-review-sentiment-analyzer/
│
├── app.py
├── train.py
├── train_distilbert.py
├── compare.py
│
├── src/
│   ├── preprocess.py
│   ├── sentiment_model.py
│   ├── transformer_models.py
│   ├── feature_extraction.py
│   ├── llm.py
│   ├── prompts.py
│   ├── prompt_analyzer.py
│   └── eval_data.py
│
├── tests/
├── data/
├── models/
├── results/
│
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/priyaraut222/customer-review-sentiment-analyzer.git

cd customer-review-sentiment-analyzer
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will be accessible locally at:

`http://localhost:8501`

### 5. Train the Models (Optional)

Train the classical ML baseline:

```bash
python train.py
```

Fine-tune DistilBERT:

```bash
python train_distilbert.py --data data/Amazon_Reviews.csv --text-col "Review Text" --rating-col "Rating" --max-samples 2000 --epochs 2 --batch-size 8
```

## Dataset & Evaluation

The classical baseline was evaluated using **21,055 usable Amazon customer reviews**.

The DistilBERT model was fine-tuned on a 2,000-review subset, consisting of 1,600 training samples and 400 held-out samples.

The project also includes a hand-labelled evaluation set of 40 reviews, including adversarial examples, for prompt-based experimentation.

## Limitations & Future Improvements

- Improve neutral sentiment classification through better class balancing and additional labelled data.
- Expand evaluation across the complete 40-review dataset.
- Improve aspect extraction to handle synonyms, plural forms, and multiple opinions within a sentence.
- Optimize LLM inference latency and structured-output consistency.
- Extend deployment and testing for production-scale usage.

