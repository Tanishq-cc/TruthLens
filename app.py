import os
import pickle
import pandas as pd
import streamlit as st

MODEL_PATH = "model/best_model.pkl"
RESULTS_PATH = "model/model_comparison.csv"

st.set_page_config(
    page_title="TruthLens | Fake News Detection",
    page_icon="📰",
    layout="wide",
)

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1150px;
}
.hero {
    padding: 1.2rem 1.4rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #111827, #1f2937);
    border: 1px solid rgba(255,255,255,.08);
    margin-bottom: 1.2rem;
}
.hero h1 {
    margin: 0;
    font-size: 3rem;
}
.hero p {
    margin: .45rem 0 0;
    color: #cbd5e1;
    font-size: 1.05rem;
}
.result-card {
    padding: 1.2rem 1.3rem;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,.12);
    background: rgba(255,255,255,.04);
}
.small-muted {
    color: #94a3b8;
    font-size: .9rem;
}
.feature-box {
    padding: .7rem .9rem;
    border-radius: 10px;
    background: rgba(255,255,255,.05);
    margin-bottom: .45rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- Load model ----------
if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run `python3 train_model.py` first.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1>📰 TruthLens</h1>
    <p>AI-powered fake news classification using NLP and Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About the project")
    st.write(
        "TruthLens uses TF-IDF text features and compares three supervised "
        "learning models to classify news as Fake or Real."
    )
    st.divider()
    st.write("**Final model:** Linear SVM")
    st.write("**Dataset:** WELFake")
    st.write("**Task:** Binary text classification")
    st.divider()
    st.caption(
        "Educational project. A prediction is not independent fact-checking."
    )

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["🔍 Analyze", "📊 Model Performance", "🧠 How It Works"])

# ---------- Analyze ----------
with tab1:
    st.subheader("Analyze a news article")
    st.write("Enter the title and article text, then let the trained model classify it.")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Try Real-News Example", use_container_width=True):
            st.session_state["title"] = (
                "Federal Reserve keeps interest rates unchanged"
            )
            st.session_state["article"] = (
                "The Federal Reserve announced Wednesday that it would keep its "
                "benchmark interest rate unchanged. Officials said future decisions "
                "will depend on incoming economic data, while noting that inflation "
                "has continued to move closer to its long-term target."
            )

    with col2:
        if st.button("Try Fake-News Example", use_container_width=True):
            st.session_state["title"] = "Scientists discover unlimited phone charging"
            st.session_state["article"] = (
                "Scientists have reportedly discovered a secret technology that "
                "allows ordinary mobile phones to charge themselves completely "
                "using energy from the air. The claim says the invention needs no "
                "electricity, batteries, or charging cables, but provides no "
                "independent testing or verifiable laboratory details."
            )

    title = st.text_input(
        "News Title",
        key="title",
        placeholder="Enter the headline..."
    )

    article = st.text_area(
        "News Article",
        key="article",
        height=260,
        placeholder="Paste the full news article here..."
    )

    analyze = st.button("🔍 Analyze Article", type="primary", use_container_width=True)

    if analyze:
        combined = f"{title} {article}".strip()

        if len(combined) < 30:
            st.warning("Please enter a longer title/article so the model has enough text to analyze.")
        else:
            prediction = int(model.predict([combined])[0])

            # LinearSVC exposes decision_function.
            score = float(model.decision_function([combined])[0])
            confidence = 1 / (1 + abs(score) ** -1) if score != 0 else 0.5
            confidence = min(max(confidence, 0.5), 0.999)

            # 0 = Fake, 1 = Real for WELFake.
            if prediction == 0:
                label = "POTENTIALLY FAKE"
                emoji = "⚠️"
                st.error(f"{emoji} {label}")
            else:
                label = "POTENTIALLY REAL"
                emoji = "✅"
                st.success(f"{emoji} {label}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", label)
            c2.metric("Model", "Linear SVM")
            c3.metric("Model confidence", f"{confidence * 100:.1f}%")

            st.progress(confidence)

            st.caption(
                "Confidence is derived from the classifier's decision score. "
                "It should not be interpreted as a probability or proof of truth."
            )

            with st.expander("🧠 See how the prediction was made"):
                st.write(
                    "The article was converted into TF-IDF features and passed "
                    "through the trained Linear SVM classifier."
                )
                st.write(f"Raw decision score: `{score:.4f}`")
                st.write(
                    "The model learns statistical language patterns from the "
                    "training data; it does not verify sources or facts."
                )

# ---------- Performance ----------
with tab2:
    st.subheader("Model Performance")
    st.write("Three baseline models were trained and evaluated on the same stratified test split.")

    if os.path.exists(RESULTS_PATH):
        results = pd.read_csv(RESULTS_PATH)
        display = results.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            if col in display.columns:
                display[col] = (display[col] * 100).round(2).astype(str) + "%"

        st.dataframe(display, use_container_width=True, hide_index=True)

        if "Accuracy" in results.columns:
            chart_df = results.set_index("Model")[["Accuracy", "F1 Score"]]
            st.bar_chart(chart_df)

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        image = "assets/class_distribution.png"
        if os.path.exists(image):
            st.image(image, caption="Dataset class distribution", use_container_width=True)

    with c2:
        image = "assets/confusion_matrix.png"
        if os.path.exists(image):
            st.image(image, caption="Confusion matrix of the best model", use_container_width=True)

# ---------- How it works ----------
with tab3:
    st.subheader("How TruthLens works")

    steps = [
        ("1", "Collect", "Use labelled Fake/Real news articles from the WELFake dataset."),
        ("2", "Clean", "Remove URLs, punctuation and unnecessary whitespace and normalize text."),
        ("3", "TF-IDF", "Convert news text into numerical features based on word importance."),
        ("4", "Train", "Compare Logistic Regression, Naive Bayes and Linear SVM."),
        ("5", "Evaluate", "Use Accuracy, Precision, Recall and F1-score."),
        ("6", "Deploy", "Save the best model and use it inside this Streamlit application."),
    ]

    for num, name, description in steps:
        st.markdown(
            f"**{num}. {name}**  \n"
            f"{description}"
        )
        st.divider()

    st.info(
        "Key limitation: TruthLens is a pattern-based classifier. It cannot "
        "independently verify whether a claim is factually true."
    )
