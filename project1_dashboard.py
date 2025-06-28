import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import joblib
import unicodedata
import regex as re
from underthesea import word_tokenize

# Load preprocessing dictionaries
def _load_txt(fn):
    with open(fn, encoding="utf8") as f:
        return [l.strip() for l in f if l.strip()]
def _load_dict(fn):
    d = {}
    with open(fn, encoding="utf8") as f:
        for line in f:
            if line.strip():
                k, v = line.rstrip("\n").split("\t", 1)
                d[k] = v
    return d

DATA_PATH = "data/"
stopwords = set(_load_txt(DATA_PATH + "vietnamese-stopwords.txt"))
wrong_words = set(_load_txt(DATA_PATH + "wrong-word.txt"))
teencode_dict = _load_dict(DATA_PATH + "teencode.txt")
emojicon_dict = _load_dict(DATA_PATH + "emojicon.txt")
eng2vn_dict = _load_dict(DATA_PATH + "english-vnmese.txt")
pos_words = _load_txt(DATA_PATH + "positive_VN.txt")
neg_words = _load_txt(DATA_PATH + "negative_VN.txt")
NEGATION_WORDS = ["không", "chưa", "chẳng", "chả"]

# Regex patterns
PUNCT_RE = re.compile(r"[^\p{L}\p{N}\s]", flags=re.UNICODE)
MULTI_RE = re.compile(r"\s+")
EMOJI_RE = re.compile("|".join(map(re.escape, emojicon_dict)))
TEEN_RE = re.compile(r"\b(" + "|".join(map(re.escape, teencode_dict)) + r")\b")
ENG_RE = re.compile(r"\b(" + "|".join(map(re.escape, eng2vn_dict)) + r")\b", flags=re.IGNORECASE)
REP_CHAR_RE = re.compile(r"(\p{L})\1{2,}", flags=re.UNICODE)

# Preprocessing functions
def normalize_repeated_characters(txt: str, max_repeat: int = 2) -> str:
    return REP_CHAR_RE.sub(lambda m: m.group(1) * max_repeat, txt)

def _join_prefixes(txt: str) -> str:
    for p in NEGATION_WORDS:
        txt = re.sub(rf"\b{p}\s+(\w+)", rf"{p}_\1", txt)
    return txt

def clean_vn(text: str) -> str:
    if not isinstance(text, str): text = ""
    txt = unicodedata.normalize("NFC", text.lower())
    txt = EMOJI_RE.sub(" ", txt)
    txt = TEEN_RE.sub(lambda m: teencode_dict[m.group(0)], txt)
    txt = ENG_RE.sub(lambda m: eng2vn_dict[m.group(0).lower()], txt)
    txt = normalize_repeated_characters(txt)
    txt = _join_prefixes(txt)
    txt = PUNCT_RE.sub(" ", txt)
    txt = MULTI_RE.sub(" ", txt).strip()
    tokens = [tok for tok in word_tokenize(txt, format="text").split() if tok not in stopwords and tok not in wrong_words and len(tok) > 1]
    return " ".join(tokens)

def join_negations(text, pos_lexicon, neg_lexicon):
    for neg in NEGATION_WORDS:
        for word in pos_lexicon:
            text = re.sub(rf"{neg}\s+{word}", f"{neg}_{word}", text)
        for word in neg_lexicon:
            text = re.sub(rf"{neg}\s+{word}", f"{neg}_{word}", text)
    return text

# Page config
st.set_page_config(page_title="ITviec Review Analyzer", layout="wide")

# Load artifacts
@st.cache_data(show_spinner="Loading data & models...")
def load_artifacts():
    df = pd.read_csv("outputs/clean_reviews.csv")
    cluster_df = pd.read_csv("outputs/company_clusters_lda.csv")
    try:
        vectorizer = joblib.load("outputs/tfidf_vectorizer.joblib")
        xgb_model = joblib.load("outputs/xgboost_sentiment_model.joblib")
    except Exception as e:
        st.error(f"Failed to load XGBoost model or vectorizer: {e}")
        vectorizer, xgb_model = None, None
    results_df = pd.read_csv("outputs/model_results.csv") if os.path.exists("outputs/model_results.csv") else None
    if results_df is not None:
        results_df = results_df[results_df["Model"] != "phoBERT"]
    lda_topics = pd.read_json("outputs/lda_topics.json") if os.path.exists("outputs/lda_topics.json") else None
    cluster_terms = pd.read_json("outputs/cluster_terms.json") if os.path.exists("outputs/cluster_terms.json") else None
    return df, cluster_df, vectorizer, xgb_model, results_df, lda_topics, cluster_terms

try:
    reviews_df, cluster_df, vectorizer, xgb_model, results_df, lda_topics, cluster_terms = load_artifacts()
except Exception as e:
    st.error(f"Load error: {e}")
    st.stop()

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Sidebar navigation
page = st.sidebar.radio(
    "📑 Select a page",
    ("📝 Sentiment & Company Explorer", "📊 Project Results")
)

st.sidebar.markdown("---")
st.sidebar.markdown("📌 **Group information:**")
st.sidebar.write("1. Yan Shih Siang")  
st.sidebar.write("• Email: ssyan110@gmail.com")  
st.sidebar.write("2. Phạm Tiến Triển ")  
st.sidebar.write("• Email: Phamtrien0211@gmail.com")

# Page 1: Sentiment & Company Explorer
if page.startswith("📝"):
    st.title("📝 ITviec Sentiment & Company Explorer")
    tab1, tab2 = st.tabs(["Sentence Sentiment", "Company View"])

    with tab1:
        st.subheader("Predict Sentiment of a Sentence")
        user_text = st.text_area("Enter a Vietnamese sentence to analyze sentiment", "")
        if st.button("Analyze Sentiment"):
            if user_text.strip():
                if vectorizer and xgb_model:
                    # Preprocess user input
                    cleaned_text = clean_vn(user_text)
                    cleaned_text = join_negations(cleaned_text, pos_words, neg_words)
                    # Vectorize and predict
                    text_vec = vectorizer.transform([cleaned_text])
                    pred = xgb_model.predict(text_vec)[0]
                    st.success(f"**Predicted Sentiment:** {LABEL_MAP[pred]}")
                else:
                    st.warning("XGBoost model or vectorizer not available. Check outputs/tfidf_vectorizer.joblib and outputs/xgboost_sentiment_model.joblib.")
            else:
                st.warning("Please enter a sentence.")

    with tab2:
        st.subheader("Analyze Company Reviews")
        company_options = sorted(reviews_df["Company Name"].unique())
        company_name = st.selectbox("Select a company", company_options)
        if company_name:
            cdata = reviews_df[reviews_df["Company Name"] == company_name]
            sentiments = cdata["sentiment"].value_counts().to_dict()
            st.markdown(f"**Sentiment Distribution:** {sentiments}")
            major_sent = cdata["sentiment"].mode().iloc[0]
            st.info(f"**Overall Company Sentiment:** {major_sent.title()}")

            # Word Cloud
            text = " ".join(cdata["clean_review"])
            wc = WordCloud(width=800, height=300, background_color="white").generate(text)
            st.image(wc.to_array(), caption="Word Cloud from Company Reviews", use_container_width=True)

            # Cluster Assignment
            cluster_row = cluster_df[cluster_df["Company Name"] == company_name]
            if not cluster_row.empty:
                cluster_id = cluster_row["cluster"].values[0]
                st.markdown(f"**Company Cluster:** {cluster_id}")
                if cluster_terms is not None and cluster_id in cluster_terms:
                    st.markdown(f"**Cluster Terms:** {', '.join(cluster_terms[cluster_id])}")
            else:
                st.warning("Company not found in cluster results.")

# Page 2: Project Results
elif page.startswith("📊"):
    st.title("📊 Project Results & Visualizations")
    st.markdown("#### Select a result below to inspect:")

    file_labels = {
        "Compare All Models (Table)": None,
        "Model Performance Chart": "ModelPerformanceChart.png",
        "PCA: Clustering Visualization": "PCA.png",
        "Confusion Matrix: KNN": "confusionMatrix_KNN.png",
        "Confusion Matrix: Logistic Regression": "confusionMatrix_LogisticRegression.png",
        "Confusion Matrix: Naive Bayes": "confusionMatrix_NaiveBayes.png",
        "Confusion Matrix: Random Forest": "confusionMatrix_RandomForest.png",
        "Confusion Matrix: XGBoost": "confusionMatrix_XGBoost.png",
        "Word Cloud: Cluster 0": "WC_C0.png",
        "Word Cloud: Cluster 1": "WC_C1.png",
        "Word Cloud: Cluster 2": "WC_C2.png"
    }

    available_files = [f for f in os.listdir("outputs") if f.endswith(".png")]
    options = [label for label, fname in file_labels.items() if fname is None or fname in available_files]

    selected = st.selectbox("Which result do you want to view?", options, index=0)

    if selected == "Compare All Models (Table)":
        st.subheader(selected)
        if results_df is not None:
            sorted_df = results_df.sort_values("F1", ascending=False).reset_index(drop=True)
            st.dataframe(
                sorted_df.style.format({
                    "Accuracy": "{:.3f}",
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1": "{:.3f}"
                }),
                use_container_width=True
            )
            st.caption("All models compared on Accuracy, Precision, Recall, and F1 (sorted by F1 Score).")
            st.markdown(
                """
                **Insights:**  
                - **XGBoost**: Top performer (highest Accuracy & F1), best for deployment.  
                - **SVM** and **Logistic Regression**: Strong but slightly below XGBoost.  
                - **Random Forest** and **Naive Bayes**: Moderate performance.  
                - **KNN**: Poor performance, not suitable.
                """
            )
        else:
            st.warning("No results table found (model_results.csv).")
    else:
        img_path = os.path.join("outputs", file_labels[selected])
        st.subheader(selected)
        st.image(img_path, use_container_width=True)

        if selected == "Model Performance Chart":
            st.caption("Compare Accuracy & F1 for each model.")
            st.markdown("**Insights:** XGBoost leads, followed by SVM and Logistic Regression. KNN performs poorly.")
        elif selected == "Confusion Matrix: KNN":
            st.caption("Diagonal = correct, off-diagonal = mistakes.")
            st.info("**KNN**: Struggles with negative/neutral reviews, often misclassifies as 'Neutral'.")
        elif selected == "Confusion Matrix: Logistic Regression":
            st.caption("Diagonal = correct, off-diagonal = mistakes.")
            st.info("**Logistic Regression**: Balanced, good at classifying all classes.")
        elif selected == "Confusion Matrix: Naive Bayes":
            st.caption("Diagonal = correct, off-diagonal = mistakes.")
            st.info("**Naive Bayes**: Over-predicts 'Positive', poor for class balance.")
        elif selected == "Confusion Matrix: Random Forest":
            st.caption("Diagonal = correct, off-diagonal = mistakes.")
            st.info("**Random Forest**: Good for 'Positive' and 'Neutral', mistakes 'Negative' as 'Positive'.")
        elif selected == "Confusion Matrix: XGBoost":
            st.caption("Diagonal = correct, off-diagonal = mistakes.")
            st.info("**XGBoost**: Strongest at 'Positive', few false positives, some neutral/negative confusion.")
        elif selected == "PCA: Clustering Visualization":
            st.caption("PCA projection of clusters.")
            st.markdown("**Insights:** Clear separation of three clusters using LDA + KMeans.")
        elif selected == "Word Cloud: Cluster 0":
            st.caption("Frequent keywords in Cluster 0.")
            st.info("**Cluster 0**: Focus on teamwork, friendly environment (e.g., 'nhân viên', 'đội', 'thân thiện').")
        elif selected == "Word Cloud: Cluster 1":
            st.caption("Frequent keywords in Cluster 1.")
            st.info("**Cluster 1**: Emphasis on comfort, benefits, clear policies (e.g., 'thoải mái', 'chế độ').")
        elif selected == "Word Cloud: Cluster 2":
            st.caption("Frequent keywords in Cluster 2.")
            st.info("**Cluster 2**: Focus on learning, projects, growth (e.g., 'dự án', 'học hỏi').")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:gray; font-size:0.9em'>"
        "ITviec Reviews Sentiment & Clustering App · Streamlit Demo"
        "</div>", unsafe_allow_html=True)
    
