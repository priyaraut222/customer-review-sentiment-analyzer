# Customer Review Sentiment Analyzer

A feedback classifier with key feature extraction. It reads customer reviews, predicts their sentiment
(positive / neutral / negative) and identifies which product features customers talk about and how they feel
about each one.

The project answers one question from a **prompt-engineering** point of view:

> How much do prompting techniques on a Hugging Face LLM improve sentiment + feature extraction,
> compared with a classical TF-IDF baseline and a fine-tuned DistilBERT?

Shah & Anchor Kutchhi Engineering College, UG Program in Electronics and Communication (Advanced Communication Technology)

## Three ways to solve the same task

| Approach | Where | Training needed | Gives features? |
|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) + spaCy noun chunks | `src/sentiment_model.py`, `src/feature_extraction.py` | yes (seconds) | yes (rule based) |
| Fine-tuned **DistilBERT** | `src/transformer_models.py`, `train_distilbert.py` | yes (minutes on GPU) | no |
| **Prompted Hugging Face LLM** (the focus) | `src/prompts.py`, `src/prompt_analyzer.py`, `src/llm.py` | **none** | yes (LLM) |

## Prompt-engineering techniques implemented

Each technique adds one idea on top of the previous one, so `compare.py` can show what each one buys you.

| Technique | Idea | Code |
|---|---|---|
| `zero_shot` | Naive baseline: a bare question | `prompts.zero_shot` |
| `role` | Persona (system prompt), `<review>` delimiters, one-word output format, **prompt-injection guard** | `prompts.role` |
| `few_shot` | Worked examples as earlier chat turns (incl. a mixed review and an injection attempt) | `prompts.few_shot` |
| `cot` | Chain-of-thought: list feature opinions, weigh them, end with `Final answer: <label>` | `prompts.cot` |
| `self_consistency` | Sample the CoT several times (temperature 0.7), **majority vote**; vote share = confidence | `PromptAnalyzer.self_consistency` |
| `structured` | JSON schema + one-shot example + **response priming** (`{`) + **repair loop** that feeds the parse error back to the model | `PromptAnalyzer.structured` |
| `chain` | **Prompt chaining**: extract features, judge each feature, then summarise using the earlier outputs | `PromptAnalyzer.chain` |
| `few_shot_noguard` | Ablation: few-shot with the injection-guard sentence removed | `compare.py` |

All prompts live in `src/prompts.py` as plain text, so they are easy to read, edit and quote in a report.

## Setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
The first run downloads model weights from the Hugging Face Hub, so it needs internet access.
A 1.5B model needs roughly 6 GB RAM on CPU. For small machines use `Qwen/Qwen2.5-0.5B-Instruct`;
the free Google Colab GPU runs the default comfortably.

## Run order
```bash
# 1. Baseline on a REAL dataset (the bundled sample data is synthetic, see "Limitations")
python train.py --data path/to/reviews.csv --text-col Text --rating-col Score

# 2. Fine-tune DistilBERT (CPU: add --max-samples 4000; Colab GPU: use the full data)
python train_distilbert.py --data path/to/reviews.csv --text-col Text --rating-col Score --max-samples 4000

# 3. Compare everything on the hand-labelled eval set (writes results/)
python compare.py                                  # all methods
python compare.py --llm Qwen/Qwen2.5-0.5B-Instruct # smaller/faster model
python compare.py --methods baseline,zero_shot,few_shot,cot --limit 12   # quick smoke run

# 4. Explore interactively
streamlit run app.py        # the "Prompt Lab" tab shows the exact prompt, raw output and parsed result
```
Switch the default LLM for everything with an environment variable: `HF_LLM=Qwen/Qwen2.5-0.5B-Instruct`.
Any instruction-tuned causal LM with a chat template works (Qwen2.5, Llama 3.x Instruct, Phi-3, Gemma-it, ...).

## What `compare.py` measures (on `data/eval_reviews.csv`)
`accuracy`, `macro_f1` (overall sentiment), `adversarial_acc` (accuracy on the 4 prompt-injection reviews),
`valid_rate` (share of replies that followed the requested format), `aspect_f1` and `aspect_sent_acc`
(feature extraction and per-feature sentiment, only for methods that output features), `sec_per_review`.
It also writes per-review predictions and the raw LLM text (`results/predictions.csv`, `results/raw_outputs.jsonl`)
for error analysis: read the wrong answers, change the prompt, re-run. That loop is the prompt-engineering work.

## Tests
```bash
python -m pytest tests -q
```
The 46 tests run **offline**: a scripted `MockLLM` checks the prompt construction, output parsers,
self-consistency voting, JSON repair loop, prompt chaining and metrics; a tiny randomly initialised DistilBERT
checks that the fine-tuning code trains, saves, reloads and predicts.

## Limitations (please read before reporting numbers)
- **The sample training data is synthetic and templated.** The baseline scores ~100% on it, which means nothing.
  Train on a real dataset (Amazon / Yelp reviews from Kaggle) and report those metrics.
- **The evaluation set is small** (40 reviews: 13 positive, 14 negative, 13 neutral, 4 of them adversarial) and was
  written and labelled by hand, so labels (especially "neutral") are partly subjective. Differences of a few
  points between methods are within noise. Extend `src/eval_data.py` or supply your own `--eval` CSV for a stronger study.
- **Feature matching is fuzzy string matching** (substring, simple trailing-s plurals); an extracted
  "battery life" matches gold "battery", but "batteries" does not match "battery".
- **Latency depends entirely on your hardware**; compare it across methods, not against other machines.
- LLM outputs are not guaranteed to be valid; `valid_rate` reports how often they were. Greedy decoding is
  deterministic for a given model and library version, but results can differ across versions.
- Few-shot examples in `src/prompts.py` are disjoint from the eval set (checked); keep it that way if you edit them.
- Aspect sentiment in the classical baseline is computed per sentence, so a sentence with mixed opinions
  gets one label for all its features. The LLM `structured` and `chain` techniques are not limited this way.

## Project structure
```
├── app.py                        # Streamlit: single review, bulk CSV, Prompt Lab, model comparison
├── train.py                      # baseline: TF-IDF + Logistic Regression
├── train_distilbert.py           # fine-tune DistilBERT (Hugging Face Transformers)
├── compare.py                    # evaluates baseline / DistilBERT / all prompt techniques
├── data/
│   ├── sample_reviews.csv        # synthetic training demo data (replace with real data)
│   └── eval_reviews.csv          # 40 hand-labelled reviews with gold feature sentiment
├── src/
│   ├── prompts.py                # all prompt templates + output parsers  <-- prompt engineering lives here
│   ├── prompt_analyzer.py        # runs each technique (voting, repair loop, chaining)
│   ├── llm.py                    # Hugging Face chat-LLM wrapper + MockLLM for tests
│   ├── transformer_models.py     # DistilBERT fine-tuning + inference
│   ├── eval_data.py              # the eval set (python -m src.eval_data rewrites the CSV)
│   ├── preprocess.py, sentiment_model.py, feature_extraction.py, generate_sample_data.py
└── tests/                        # offline test suite
```

## SDG alignment
SDG 8 (Decent Work and Economic Growth), SDG 9 (Industry, Innovation and Infrastructure), SDG 12 (Responsible Consumption and Production)

## Team
Member 1 (Data and Modelling), Member 2 (NLP and Feature Extraction), Member 3 (Integration and Documentation)

## License
MIT

## Running the app, comparison and tests

Install dependencies:

    pip install -r requirements.txt

Start the Streamlit app:

    streamlit run app.py

Compare the TF-IDF baseline and DistilBERT on the evaluation set:

    python compare.py

Run the tests:

    python -m pytest tests
