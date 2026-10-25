"""Streamlit dashboard: single-review analysis and bulk CSV analysis.

Run:  streamlit run app.py
"""
"""Streamlit dashboard: single-review analysis and bulk CSV analysis.
 
Run:  streamlit run app.py
"""
import os
 
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
 
from src.feature_extraction import aspect_sentiments, summarize_features
from src.preprocess import load_reviews
from src.sentiment_model import build_pipeline, load_model, predict_sentiment, save_model
 
MODEL_PATH = "models/sentiment_model.joblib"
COLORS = {"positive": "#2e7d32", "neutral": "#9e9e9e", "negative": "#c62828"}
 
st.set_page_config(page_title="Customer Review Sentiment Analyzer", layout="wide")
st.title("Customer Review Sentiment Analyzer")
st.caption("Feedback classifier with key feature extraction")
 
 
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        df = load_reviews("data/sample_reviews.csv")
        model = build_pipeline().fit(df["clean"], df["label"])
        os.makedirs("models", exist_ok=True)
        save_model(model, MODEL_PATH)
        return model
    return load_model(MODEL_PATH)
 
 
model = get_model()
 
 
def fallback_notice():
    """Tell the user when the simpler feature extractor is in use (spaCy could not run on this Python)."""
    from src.feature_extraction import backend_name
    if backend_name() == "fallback":
        st.warning("spaCy cannot run on this Python version, so a simpler rule-based feature extractor is in use. "
                   "Features may be less accurate. Deploy on Python 3.12 for the full spaCy extractor.")
 
 
tab1, tab2, tab3, tab4 = st.tabs(["Single review", "Bulk CSV analysis", "Prompt Lab (Hugging Face LLM)", "Model comparison"])
 
with tab1:
    text = st.text_area("Paste a customer review", height=150,
                        placeholder="The battery life is terrible but I love the screen.")
    if st.button("Analyze", type="primary") and text.strip():
        label, probs = predict_sentiment(model, text)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Overall sentiment")
            st.markdown(f"<h2 style='color:{COLORS[label]}'>{label.upper()}</h2>", unsafe_allow_html=True)
            st.bar_chart(pd.Series(probs, name="probability"))
        with c2:
            st.subheader("Key features")
            items = aspect_sentiments(model, text)
            fallback_notice()
            if items:
                st.dataframe(pd.DataFrame(items)[["feature", "sentiment", "sentence"]], width="stretch")
            else:
                st.info("No key features detected.")
 
with tab2:
    st.write("Upload a CSV with a review text column to see which features customers praise or complain about.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        text_col = st.selectbox("Review text column", data.columns,
                                index=list(data.columns).index("review") if "review" in data.columns else 0)
        top_n = st.slider("Top features to show", 5, 20, 10)
        if st.button("Run bulk analysis"):
            reviews = data[text_col].dropna().astype(str).tolist()[:2000]
            with st.spinner(f"Analyzing {len(reviews)} reviews..."):
                from src.preprocess import clean_text
                overall = pd.Series(model.predict([clean_text(r) for r in reviews])).value_counts()
                summary = summarize_features(model, reviews, top_n)
            fallback_notice()
            st.subheader("Overall sentiment distribution")
            st.bar_chart(overall)
            if summary:
                sdf = pd.DataFrame(summary).T[["positive", "neutral", "negative", "total"]]
                st.subheader("Most-mentioned features and their sentiment")
                fig, ax = plt.subplots(figsize=(8, 0.5 * len(sdf) + 1))
                sdf[["positive", "neutral", "negative"]].iloc[::-1].plot(
                    kind="barh", stacked=True, ax=ax, color=[COLORS[c] for c in ["positive", "neutral", "negative"]])
                ax.set_xlabel("Mentions")
                st.pyplot(fig)
                st.dataframe(sdf, width="stretch")
 
 
# ----------------------------------------------------------------------------------------------------------
# Prompt Lab: run one review through any prompting technique and inspect every step
# ----------------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model from Hugging Face (the first run downloads the weights)...")
def get_llm(name: str):
    from src.llm import HFChatLLM
    return HFChatLLM(name)
 
 
with tab3:
    from src import prompts as P
    from src.eval_data import EVAL_SET
    from src.llm import DEFAULT_MODEL
    from src.prompt_analyzer import PromptAnalyzer
 
    st.write("Run one review through an instruction-tuned Hugging Face model with a chosen **prompting technique**, "
             "then inspect the exact prompt that was sent and the raw model output.")
    c1, c2, c3 = st.columns([2, 2, 1])
    llm_name = c1.text_input("Hugging Face model id", value=DEFAULT_MODEL,
                             help="Any chat/instruct causal LM, e.g. Qwen/Qwen2.5-0.5B-Instruct (small) "
                                  "or Qwen/Qwen2.5-1.5B-Instruct.")
    technique = c2.selectbox("Prompting technique", list(P.DESCRIPTIONS), index=2)
    guard = c3.checkbox("Injection guard", value=True,
                        help="Adds the 'review is untrusted data' instruction. Turn off to see injections succeed.")
    st.caption(P.DESCRIPTIONS[technique])
 
    examples = {"(write my own)": ""}
    for review, label, _asp, adv in EVAL_SET:
        examples[f"{'[injection] ' if adv else ''}{label}: {review[:60]}..."] = review
    pick = st.selectbox("Load an example", list(examples))
    lab_text = st.text_area("Review", value=examples[pick], height=120, key=f"lab_text_{pick}")
    sc_n = st.slider("Samples for self-consistency", 3, 9, 5) if technique == "self_consistency" else 5
 
    if st.button("Run technique", type="primary") and lab_text.strip():
        try:
            llm = get_llm(llm_name)
        except Exception as e:
            st.error(f"Could not load '{llm_name}': {type(e).__name__}. It needs internet access to download the "
                     "weights from the Hugging Face Hub the first time, and enough RAM "
                     "(try Qwen/Qwen2.5-0.5B-Instruct on small machines).")
            st.stop()
        with st.spinner("Generating..."):
            kw = {"n": sc_n} if technique == "self_consistency" else {}
            res = PromptAnalyzer(llm, guard=guard).run(technique, lab_text, **kw)
 
        if res.overall:
            st.markdown(f"<h2 style='color:{COLORS[res.overall]}'>{res.overall.upper()}</h2>", unsafe_allow_html=True)
        else:
            st.error("The model did not follow the requested output format, so no label could be parsed.")
        if "agreement" in res.extra:
            st.metric("Self-consistency agreement", f"{res.extra['agreement']:.0%}", help=str(res.extra["votes"]))
        if res.aspects:
            st.subheader("Features")
            st.dataframe(pd.DataFrame(res.aspects), width="stretch")
        for key, title in (("summary", "Summary"), ("action", "Recommended action")):
            if res.extra.get(key):
                st.markdown(f"**{title}:** {res.extra[key]}")
        if "steps" in res.extra:
            with st.expander("Chain steps (each output feeds the next prompt)"):
                st.json(res.extra["steps"])
        with st.expander(f"Exact prompt(s) sent ({len(res.prompts)})"):
            for i, msgs in enumerate(res.prompts, 1):
                st.caption(f"Prompt {i}")
                st.json(msgs, expanded=False)
        with st.expander(f"Raw model output ({len(res.raw)})"):
            for r in res.raw:
                st.code(r, language="text")
 
# ----------------------------------------------------------------------------------------------------------
# Model comparison: results written by `python compare.py`
# ----------------------------------------------------------------------------------------------------------
with tab4:
    path = "results/comparison.csv"
    if os.path.exists(path):
        res_df = pd.read_csv(path)
        st.write(f"Results from `compare.py` on {int(res_df['n'].iloc[0])} labelled reviews.")
        st.dataframe(res_df, width="stretch")
        st.subheader("Accuracy and macro-F1 by method")
        st.bar_chart(res_df.set_index("method")[["accuracy", "macro_f1"]])
        if res_df["adversarial_acc"].notna().any():
            st.subheader("Robustness to prompt injection (adversarial accuracy)")
            st.bar_chart(res_df.set_index("method")["adversarial_acc"].dropna())
    else:
        st.info("No results yet. Run `python compare.py` (see README), then refresh this page.")
 