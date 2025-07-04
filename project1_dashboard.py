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

def analyze_company_feedback(company_data, company_name, cluster_id, all_data):
    """
    Analyze what a company is doing well and what needs improvement,
    then provide actionable suggestions.
    """
    feedback = {
        "strengths": [],
        "weaknesses": [],
        "suggestions": []
    }
    
    # Calculate metrics
    total_reviews = len(company_data)
    sentiment_dist = company_data["sentiment"].value_counts(normalize=True)
    avg_rating = company_data["Rating"].mean()
    recommend_rate = (company_data["Recommend?"] == "Yes").mean() * 100
    
    # Rating subscores
    subscores = {
        "Salary & benefits": company_data["Salary & benefits"].mean(),
        "Training & learning": company_data["Training & learning"].mean(), 
        "Management cares about me": company_data["Management cares about me"].mean(),
        "Culture & fun": company_data["Culture & fun"].mean(),
        "Office & workspace": company_data["Office & workspace"].mean()
    }
    
    # Industry averages for comparison
    industry_avg = {
        "rating": all_data["Rating"].mean(),
        "recommend": (all_data["Recommend?"] == "Yes").mean() * 100,
        "salary": all_data["Salary & benefits"].mean(),
        "training": all_data["Training & learning"].mean(),
        "management": all_data["Management cares about me"].mean(),
        "culture": all_data["Culture & fun"].mean(),
        "office": all_data["Office & workspace"].mean()
    }
    
    # ANALYZE STRENGTHS
    if sentiment_dist.get("positive", 0) > 0.6:
        feedback["strengths"].append(f"🎉 Strong positive sentiment ({sentiment_dist.get('positive', 0):.1%} of reviews)")
    
    if avg_rating > industry_avg["rating"] + 0.3:
        feedback["strengths"].append(f"⭐ Above-average overall rating ({avg_rating:.1f}/5.0 vs industry {industry_avg['rating']:.1f})")
    
    if recommend_rate > industry_avg["recommend"] + 10:
        feedback["strengths"].append(f"👍 High recommendation rate ({recommend_rate:.0f}% vs industry {industry_avg['recommend']:.0f}%)")
    
    # Check individual subscores
    for area, score in subscores.items():
        if area == "Salary & benefits" and score > industry_avg["salary"] + 0.3:
            feedback["strengths"].append(f"💰 Excellent {area} ({score:.1f}/5.0)")
        elif area == "Training & learning" and score > industry_avg["training"] + 0.3:
            feedback["strengths"].append(f"📚 Strong {area} ({score:.1f}/5.0)")
        elif area == "Management cares about me" and score > industry_avg["management"] + 0.3:
            feedback["strengths"].append(f"🤝 Great {area} ({score:.1f}/5.0)")
        elif area == "Culture & fun" and score > industry_avg["culture"] + 0.3:
            feedback["strengths"].append(f"🎨 Excellent {area} ({score:.1f}/5.0)")
        elif area == "Office & workspace" and score > industry_avg["office"] + 0.3:
            feedback["strengths"].append(f"🏢 Great {area} ({score:.1f}/5.0)")
    
    # ANALYZE WEAKNESSES  
    if sentiment_dist.get("negative", 0) > 0.3:
        feedback["weaknesses"].append(f"⚠️ High negative sentiment ({sentiment_dist.get('negative', 0):.1%} of reviews)")
    
    if avg_rating < industry_avg["rating"] - 0.3:
        feedback["weaknesses"].append(f"📉 Below-average rating ({avg_rating:.1f}/5.0 vs industry {industry_avg['rating']:.1f})")
    
    if recommend_rate < industry_avg["recommend"] - 10:
        feedback["weaknesses"].append(f"👎 Low recommendation rate ({recommend_rate:.0f}% vs industry {industry_avg['recommend']:.0f}%)")
    
    # Check weak subscores
    weak_areas = []
    for area, score in subscores.items():
        if area == "Salary & benefits" and score < industry_avg["salary"] - 0.3:
            weak_areas.append(f"💸 {area} needs improvement ({score:.1f}/5.0)")
        elif area == "Training & learning" and score < industry_avg["training"] - 0.3:
            weak_areas.append(f"📖 {area} needs attention ({score:.1f}/5.0)")  
        elif area == "Management cares about me" and score < industry_avg["management"] - 0.3:
            weak_areas.append(f"👔 {area} needs work ({score:.1f}/5.0)")
        elif area == "Culture & fun" and score < industry_avg["culture"] - 0.3:
            weak_areas.append(f"🎭 {area} could be better ({score:.1f}/5.0)")
        elif area == "Office & workspace" and score < industry_avg["office"] - 0.3:
            weak_areas.append(f"🏗️ {area} needs upgrading ({score:.1f}/5.0)")
    
    feedback["weaknesses"].extend(weak_areas)
    
    # GENERATE SUGGESTIONS based on cluster and weaknesses
    if cluster_id == 0:  # Teamwork & friendly environment cluster
        if any("Management" in w for w in weak_areas):
            feedback["suggestions"].append("🤝 Focus on leadership training and team-building activities")
        if any("Culture" in w for w in weak_areas):
            feedback["suggestions"].append("🎨 Organize more team events and improve communication channels")
        feedback["suggestions"].append("👥 Leverage your strength in teamwork by creating mentorship programs")
        
    elif cluster_id == 1:  # Comfort & benefits cluster  
        if any("Salary" in w for w in weak_areas):
            feedback["suggestions"].append("💰 Review compensation packages and benchmark against market rates")
        if any("Office" in w for w in weak_areas):
            feedback["suggestions"].append("🏢 Invest in workspace improvements and modern facilities")
        feedback["suggestions"].append("🛡️ Enhance employee benefits packages (health, wellness, flexibility)")
        
    elif cluster_id == 2:  # Learning & growth cluster
        if any("Training" in w for w in weak_areas):
            feedback["suggestions"].append("📚 Develop comprehensive learning paths and skill development programs")
        if any("Management" in w for w in weak_areas):
            feedback["suggestions"].append("📈 Implement regular career progression discussions and goal setting")
        feedback["suggestions"].append("🚀 Create innovation time and project-based learning opportunities")
    
    # General suggestions based on common issues
    if sentiment_dist.get("negative", 0) > 0.3:
        feedback["suggestions"].append("🔍 Conduct exit interviews and employee surveys to identify specific pain points")
    
    if recommend_rate < 70:
        feedback["suggestions"].append("💬 Implement regular one-on-one meetings and feedback sessions")
    
    return feedback

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

st.sidebar.markdown("""
**Yêu cầu 1:** Các công ty đang nhận nhiều đánh giá (review) từ ứng viên/nhân viên đăng trên ITViec,  
dựa trên những thông tin này để phân tích cảm xúc (tích cực, tiêu cực, trung tính).

**Yêu cầu 2:** Dựa trên những thông tin từ review của ứng viên/nhân viên đăng trên ITViec để phân cụm thông tin đánh giá (Information Clustering).
""")

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
            # Additional filtering for better word cloud quality
            additional_stopwords = {
                "không", "công_ty", "công", "ty", "được", "có", "là", "và", "của", "cho", "với", "từ", "trong", "này", 
                "đó", "thì", "sẽ", "đã", "đang", "rất", "nhiều", "ở", "về", "như", "khi", "nếu", "mà", "để", "bị", 
                "những", "các", "một", "hai", "ba", "cũng", "nữa", "thêm", "khác", "chỉ", "vào", "ra", "lên", "xuống", "nhân_viên", "công_việc", "làm_việc", "văn_phòng"
            }
            
            # Clean and filter the text
            all_words = []
            for review in cdata["clean_review"].dropna():
                words = review.split()
                # Filter out additional stopwords and short words
                filtered_words = [word for word in words 
                                if word not in additional_stopwords 
                                and len(word) > 2 
                                and not word.isdigit()]
                all_words.extend(filtered_words)
            
            clean_text = " ".join(all_words)
            
            if clean_text.strip():
                wc = WordCloud(
                    width=800, 
                    height=300, 
                    background_color="white",
                    max_words=50, 
                    relative_scaling=0.5,  
                    colormap='viridis', 
                    min_font_size=10,
                    max_font_size=100
                ).generate(clean_text)
                st.image(wc.to_array(), caption="Word Cloud from Company Reviews (Filtered)", use_container_width=True)
            else:
                st.info("No sufficient text data available for word cloud generation.")

            # Cluster Assignment
            cluster_row = cluster_df[cluster_df["Company Name"] == company_name]
            if not cluster_row.empty:
                cluster_id = cluster_row["cluster"].values[0]
                st.markdown(f"**Company Cluster:** {cluster_id}")
                
                # Add clear cluster descriptions
                if cluster_id == 0:
                    st.info("**Cluster 0**: Staffs talked about teamwork, friendly environment (e.g., 'nhân viên', 'đội', 'thân thiện') the most.")
                elif cluster_id == 1:
                    st.info("**Cluster 1**: Staffs talked about comfort, benefits, clear policies (e.g., 'thoải mái', 'chế độ') the most.")
                elif cluster_id == 2:
                    st.info("**Cluster 2**: Staffs talked about learning, projects, growth (e.g., 'dự án', 'học hỏi') the most.")
                
                if cluster_terms is not None and cluster_id in cluster_terms:
                    st.markdown(f"**Cluster Terms:** {', '.join(cluster_terms[cluster_id])}")
                    
                # Company Performance Analysis & Feedback
                st.markdown("---")
                st.subheader("📊 Company Performance Analysis & Recommendations")
                
                feedback = analyze_company_feedback(cdata, company_name, cluster_id, reviews_df)
                
                # Create three columns for feedback
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 🎯 **What You're Doing Well**")
                    if feedback["strengths"]:
                        for strength in feedback["strengths"]:
                            st.success(strength)
                    else:
                        st.info("Continue working on building your strengths! 💪")
                
                with col2:
                    st.markdown("### ⚠️ **Areas for Improvement**")
                    if feedback["weaknesses"]:
                        for weakness in feedback["weaknesses"]:
                            st.warning(weakness)
                    else:
                        st.success("Great job! No major weaknesses identified! 🌟")
                
                with col3:
                    st.markdown("### 💡 **Actionable Suggestions**")
                    if feedback["suggestions"]:
                        for suggestion in feedback["suggestions"]:
                            st.info(suggestion)
                    else:
                        st.success("Keep up the excellent work! 🚀")
                
                # Additional insights
                st.markdown("---")
                st.markdown("### 📈 **Detailed Rating Breakdown**")
                rating_data = {
                    "Area": ["Salary & benefits", "Training & learning", "Management cares about me", "Culture & fun", "Office & workspace"],
                    "Your Score": [
                        cdata["Salary & benefits"].mean(),
                        cdata["Training & learning"].mean(),
                        cdata["Management cares about me"].mean(),
                        cdata["Culture & fun"].mean(),
                        cdata["Office & workspace"].mean()
                    ],
                    "Industry Avg": [
                        reviews_df["Salary & benefits"].mean(),
                        reviews_df["Training & learning"].mean(),
                        reviews_df["Management cares about me"].mean(),
                        reviews_df["Culture & fun"].mean(),
                        reviews_df["Office & workspace"].mean()
                    ]
                }
                
                import pandas as pd
                rating_df = pd.DataFrame(rating_data)
                rating_df["Difference"] = rating_df["Your Score"] - rating_df["Industry Avg"]
                rating_df["Your Score"] = rating_df["Your Score"].round(2)
                rating_df["Industry Avg"] = rating_df["Industry Avg"].round(2)
                rating_df["Difference"] = rating_df["Difference"].round(2)
                
                st.dataframe(
                    rating_df.style.format({
                        "Your Score": "{:.2f}",
                        "Industry Avg": "{:.2f}",
                        "Difference": "{:+.2f}"
                    }).applymap(
                        lambda x: 'background-color: lightgreen' if isinstance(x, (int, float)) and x > 0 else 
                                 'background-color: lightcoral' if isinstance(x, (int, float)) and x < -0.2 else '',
                        subset=['Difference']
                    ),
                    use_container_width=True
                )
                
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
