"""Streamlit app: load reviews CSV, run OpenAI sentiment, show chart."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple
import openai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# Initialize OpenAI client
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# How many rows to send per API call (keeps requests under token limits).
BATCH_SIZE = 10


def get_dataset_path() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "data", "customer_reviews_copy.csv")


def extract_first_json(text: str) -> Any:
    if not text:
        return None
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return None
    start = min(starts)
    end = text.rfind("]" if text[start] == "[" else "}")
    if end == -1 or end < start:
        return None
    return json.loads(text[start : end + 1])


@st.cache_data(show_spinner=False)
def analyze_sentiment_batch(
    model: str,
    items: Tuple[Tuple[int, str], ...],
) -> Tuple[List[float], List[str]]:
    """
    Call OpenAI for one batch of (row_id, text). Cached so identical model + texts
    skip repeat API calls (e.g. reruns, widget interaction).
    """
    items_list = list(items)
    max_chars = 2500
    lines = []
    for idx, t in items_list:
        safe = (t or "")[:max_chars].replace("\n", " ").strip()
        lines.append(f"{idx}: {safe}")

    prompt = (
        "You are a sentiment analysis engine. Clean/normalize each text, then score sentiment.\n\n"
        "Rules: score in [-1, 1]; label is positive, neutral, or negative.\n"
        'Return ONLY JSON: {"results":[{"id":<id>,"score":<float>,"label":"..."}]}\n\n'
        "Input:\n" + "\n".join(lines)
    )

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = getattr(response, "output_text", None) or str(response)
    data = extract_first_json(raw)
    if not data or "results" not in data:
        return [0.0] * len(items_list), ["neutral"] * len(items_list)

    by_id: Dict[int, Dict[str, Any]] = {}
    for r in data.get("results", []):
        if not isinstance(r, dict) or "id" not in r:
            continue
        try:
            by_id[int(r["id"])] = r
        except (TypeError, ValueError):
            continue

    scores, labels = [], []
    for idx, _ in items_list:
        r = by_id.get(idx)
        if not r:
            scores.append(0.0)
            labels.append("neutral")
        else:
            scores.append(float(r.get("score", 0.0)))
            labels.append(str(r.get("label", "neutral")))
    return scores, labels


@st.dialog("Analyze which column?")
def show_column_picker_dialog(candidate_cols: List[str]) -> None:
    default_i = candidate_cols.index("SUMMARY") if "SUMMARY" in candidate_cols else 0
    col = st.selectbox(
        "Analyze which column?",
        candidate_cols,
        index=default_i,
        key="dialog_text_column",
    )
    if st.button("Run Sentiment Analysis"):
        st.session_state["selected_text_column"] = col
        st.session_state["run_sentiment_now"] = True
        st.session_state["show_column_dialog"] = False
        st.rerun()


st.title(" GenAI Sentiment Analysis 🤖")
st.write(
    "I can help you with sentiment analysis for your products reviews, comments and more."
)

col_load, col_analyze = st.columns(2)
with col_load:
    if st.button(" 📥 Load Dataset"):
        try:
            st.session_state["df"] = pd.read_csv(get_dataset_path()).head(10)
            st.session_state["sentiment_ready"] = False
            st.session_state["show_column_dialog"] = False
            st.success("Dataset ingested successfully")
        except FileNotFoundError:
            st.error("Dataset not found. Please check the file path.")
with col_analyze:
    if st.button(" 🧠 Analyze Sentiment"):
        if "df" in st.session_state:
            st.session_state["show_column_dialog"] = True
            st.session_state["run_sentiment_now"] = False
        else:
            st.warning("Please ingest the dataset first.")

if "df" not in st.session_state:
    st.stop()

st.session_state.setdefault("sentiment_ready", False)
st.session_state.setdefault("show_column_dialog", False)

df = st.session_state["df"]

st.subheader(" 🔍 Filter by Product")
product_filter = st.selectbox(
    "Select a product",
    ["All Products"] + df["PRODUCT"].unique().tolist(),
)
st.subheader(" 📁 Dataset Preview")
if product_filter != "All Products":
    filtered_df = df[df["PRODUCT"] == product_filter]
    st.success(f"Filtered by {product_filter}")
else:
    filtered_df = df
st.dataframe(filtered_df)

if "SENTIMENT_SCORE" not in df.columns:
    df["SENTIMENT_SCORE"] = 0.0
if "SENTIMENT_LABEL" not in df.columns:
    df["SENTIMENT_LABEL"] = None

candidate_cols = [
    c for c in df.columns if c not in ("PRODUCT", "SENTIMENT_SCORE", "SENTIMENT_LABEL")
]
if not candidate_cols:
    st.error("No text columns found to analyze.")
    st.stop()

if st.session_state["show_column_dialog"]:
    show_column_picker_dialog(candidate_cols)

run_now = st.session_state.pop("run_sentiment_now", False)
text_column = st.session_state.get("selected_text_column")
model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

if run_now and text_column:
    with st.spinner("Analyzing sentiment..."):
        texts = df[text_column].fillna("").astype(str).tolist()
        scores_out: List[float] = [0.0] * len(texts)
        labels_out: List[str] = ["neutral"] * len(texts)
        progress = st.progress(0)
        n = len(texts)

        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            batch = [(i, texts[i]) for i in range(start, end)]
            s_batch, l_batch = analyze_sentiment_batch(model_name, tuple(batch))
            for off, (s, lab) in enumerate(zip(s_batch, l_batch)):
                pos = start + off
                scores_out[pos] = s
                labels_out[pos] = lab
            progress.progress(min(1.0, end / max(n, 1)))

        df["SENTIMENT_SCORE"] = scores_out
        df["SENTIMENT_LABEL"] = labels_out
        st.session_state["df"] = df
        st.session_state["sentiment_ready"] = True
        st.success("Sentiment analysis complete.")

if st.session_state.get("sentiment_ready"):
    st.subheader(" 📊 Sentiment chart")
    df_chart = st.session_state["df"]

    if product_filter == "All Products":
        grouped = (
            df_chart.groupby("PRODUCT")["SENTIMENT_SCORE"]
            .mean()
            .reset_index()
            .sort_values(by="SENTIMENT_SCORE", ascending=False)
        )
        st.bar_chart(grouped, x="PRODUCT", y="SENTIMENT_SCORE")
        st.caption("Average sentiment score **per product** (all rows).")
    else:
        sub = df_chart[df_chart["PRODUCT"] == product_filter].reset_index(drop=True)
        plot_df = pd.DataFrame(
            {
                "Review": [f"#{i + 1}" for i in range(len(sub))],
                "SENTIMENT_SCORE": sub["SENTIMENT_SCORE"].astype(float),
            }
        )
        st.bar_chart(plot_df, x="Review", y="SENTIMENT_SCORE")
        st.caption(
            f"Sentiment score **per review** for **{product_filter}** "
            f"({len(sub)} row{'s' if len(sub) != 1 else ''})."
        )
