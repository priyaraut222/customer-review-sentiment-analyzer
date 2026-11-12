# Customer Review Sentiment Analyzer
### NLP | Fine-Tuned Transformers | Prompt Engineering | Aspect-Based Sentiment Analysis

An end-to-end NLP application that combines classical machine learning, transformer fine-tuning, and LLM-based prompt engineering to analyse customer feedback, classify sentiment, and extract feature-level opinions.

## Overview

Customer reviews often contain opinions about multiple product features within a single piece of feedback. Understanding not just *how customers feel*, but *what they feel about*, is essential for meaningful sentiment analysis.

This project explores three approaches to solving this problem:

- **Classical Machine Learning:** TF-IDF with Logistic Regression and spaCy-based feature extraction.
- **Transformer-Based Learning:** Fine-tuning DistilBERT for sentiment classification.
- **LLM-Based Analysis:** Experimenting with Hugging Face instruction-tuned models through different prompt engineering strategies.

The central objective is to investigate how prompting techniques influence classification performance, feature extraction, output consistency, and adversarial robustness compared with conventional ML and fine-tuned transformer approaches.

## Key Features

- **Sentiment Classification:** Identifies positive, negative, and neutral customer opinions.
- **Aspect-Based Sentiment Analysis:** Extracts product and service features and determines the sentiment associated with individual aspects.
- **DistilBERT Fine-Tuning:** Implements transformer-based sentiment classification using Hugging Face Transformers.
- **Prompt Engineering Lab:** Implements and compares multiple prompting strategies, including zero-shot, role-based, few-shot, Chain-of-Thought, self-consistency, structured prompting, and prompt chaining.
- **Adversarial Evaluation:** Tests model behaviour against prompt-injection attempts and compares guarded and unguarded prompting.
- **Structured LLM Outputs:** Implements JSON validation, response parsing, and automated repair for malformed outputs.
- **Interactive Streamlit Interface:** Supports individual reviews, bulk CSV analysis, prompt experimentation, and model comparison.
- **Comprehensive Evaluation:** Measures accuracy, macro-F1, aspect-level metrics, adversarial accuracy, output validity, and inference latency.

## Approaches & Prompting Strategies

| Approach | Implementation |
|---|---|
| Classical ML | TF-IDF + Logistic Regression |
| Transformer | Fine-tuned DistilBERT |
| Zero-shot | Direct instruction-based classification |
| Role-based | Persona-driven prompting with guard instructions |
| Few-shot | Classification using illustrative examples |
| Few-shot (No Guard) | Ablation experiment without injection-guard instructions |
| Chain-of-Thought | Step-by-step reasoning for sentiment analysis |
| Self-Consistency | Multiple generations with majority voting |
| Structured | JSON schema, validation and repair loop |
| Prompt Chaining | Sequential feature extraction, sentiment assessment and summarisation |

All prompt templates are maintained separately, making experimentation and modification straightforward.

## Experimental Results

All approaches were evaluated on a hand-labelled dataset of **40 customer reviews**, including four adversarial examples.

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 70.0% | 0.689 |
| DistilBERT | 57.5% | 0.469 |
| Zero-shot | 87.5% | 0.908 |
| Few-shot | 92.5% | 0.921 |
| Few-shot (No Guard) | **95.0%** | **0.948** |
| Chain-of-Thought | 87.5% | 0.864 |
| Self-Consistency | 87.5% | 0.864 |
| Structured | 80.0% | 0.766 |
| Prompt Chaining | 92.5% | 0.924 |

**Key observations:**
- Few-shot prompting without guard instructions achieved the highest classification accuracy and macro-F1 on this evaluation set.
- Few-shot prompting with guard instructions achieved 92.5% accuracy and 100% adversarial accuracy.
- Zero-shot prompting achieved 100% adversarial accuracy with 87.5% overall accuracy.
- Structured prompting and prompt chaining additionally support aspect-level extraction and sentiment evaluation.
- Self-consistency introduced considerably higher inference latency.

*Results are preliminary and based on a small, manually labelled evaluation set. They should not be interpreted as generalisable performance benchmarks.*

## Evaluation Metrics

The comparison framework measures:

- **Accuracy & Macro-F1:** Overall sentiment classification performance.
- **Adversarial Accuracy:** Classification accuracy on prompt-injection examples.
- **Valid Rate:** Proportion of outputs following the requested format.
- **Aspect F1:** Performance of feature extraction.
- **Aspect Sentiment Accuracy:** Correctness of sentiment assigned to extracted features.
- **Inference Latency:** Average processing time per review.

Per-review predictions and raw LLM outputs are saved for error analysis and further prompt experimentation.

## Project Structure

```text
customer-review-sentiment-analyzer/
│
├── app.py                       # Streamlit application and interactive dashboard
├── train.py                     # TF-IDF + Logistic Regression baseline training
├── train_distilbert.py          # DistilBERT fine-tuning and evaluation
├── compare.py                   # Comparison of ML, transformer and prompting approaches
│
├── data/
│   ├── sample_reviews.csv        # Synthetic sample training data
│   └── eval_reviews.csv          # 40 hand-labelled evaluation reviews
│
├── src/
│   ├── prompts.py                # Prompt templates and output parsers
│   ├── prompt_analyzer.py        # Prompt execution, voting, chaining and JSON repair
│   ├── llm.py                    # Hugging Face LLM integration and MockLLM
│   ├── transformer_models.py     # DistilBERT fine-tuning and inference
│   ├── sentiment_model.py        # Classical sentiment classification
│   ├── feature_extraction.py     # Aspect and feature extraction
│   ├── preprocess.py             # Text preprocessing and rating conversion
│   ├── eval_data.py              # Hand-labelled evaluation dataset
│   └── generate_sample_data.py   # Synthetic sample data generation
│
├── tests/                        # Offline automated test suite
├── models/                       # Saved trained models
├── results/                      # Evaluation metrics, predictions and raw outputs
│
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Classical ML | Scikit-learn |
| NLP | spaCy |
| Deep Learning | PyTorch, Hugging Face Transformers |
| LLMs | Qwen2.5 Instruct |
| Prompt Engineering | Zero-shot, Few-shot, CoT, Self-Consistency, Prompt Chaining |
| Interface | Streamlit |
| Testing | Pytest |

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

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run the Application

```bash
streamlit run app.py
```

### 4. Run Experiments

Train the classical baseline:

```bash
python train.py --data path/to/reviews.csv --text-col Text --rating-col Score
```

Fine-tune DistilBERT:

```bash
python train_distilbert.py --data path/to/reviews.csv --text-col Text --rating-col Score
```

Compare different approaches:

```bash
python compare.py
```

Run the test suite:

```bash
python -m pytest tests -q
```

*The first LLM run requires internet access to download the Hugging Face model weights. An appropriate model and sufficient system memory are required.*

## Limitations & Future Scope

- Expand evaluation using larger, independently labelled datasets such as Amazon or Yelp reviews.
- Improve neutral sentiment classification and class balance.
- Enhance aspect matching to support synonyms, plural forms, and complex mixed opinions.
- Investigate prompt-injection robustness across more diverse adversarial inputs.
- Explore larger instruction-tuned models and optimise inference latency.
- Improve reproducibility through expanded benchmarking and model-version tracking.
