import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ==========================
# Load CSS
# ==========================
def load_css():
    css_file = Path("styles.css")
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ==========================
# Load Model
# ==========================
MODEL_PATH = Path("models/fraud_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(" Model file not found. Please check models/fraud_model.pkl")
    st.stop()

# ==========================
# Sidebar
# ==========================
st.sidebar.image(
    "https://img.icons8.com/color/96/bank-card-back-side.png",
    width=80
)

st.sidebar.title("Credit Card Fraud Detection")

st.sidebar.markdown("---")

st.sidebar.info(
    """
This application detects fraudulent
credit card transactions using
Machine Learning.
"""
)

st.sidebar.markdown("### 🤖 Model")
st.sidebar.success("Random Forest Classifier")

st.sidebar.markdown("### 💻 Language")
st.sidebar.write("Python")

st.sidebar.markdown("### 🚀 Framework")
st.sidebar.write("Streamlit")

# FIX 1: Sidebar live stats are populated after prediction runs.
# Use session_state so they persist across reruns without crashing on first load.
if "fraud_count" not in st.session_state:
    st.session_state.fraud_count = 0
    st.session_state.normal_count = 0
    st.session_state.total_count = 0

st.sidebar.markdown("---")
st.sidebar.caption("Developed by Prayag Sapariya")

# ==========================
# Header
# ==========================
st.markdown(
    """
<p class="main-title">
💳 Credit Card Fraud Detection Dashboard
</p>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
<p class="sub-title">
Detect fraudulent credit card transactions using Machine Learning
</p>
""",
    unsafe_allow_html=True
)

st.divider()

# ==========================
# Upload CSV
# ==========================
uploaded_file = st.file_uploader(
    "📂 Upload CSV File",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success(" Dataset Uploaded Successfully")

    # ==========================
    # Metrics
    # ==========================
    col1, col2, col3 = st.columns(3)

    col1.metric("📄 Rows", df.shape[0])
    col2.metric("📊 Columns", df.shape[1])

    if "Class" in df.columns:
        col3.metric("🚨 Fraud Cases", int(df["Class"].sum()))
    else:
        col3.metric("🚨 Fraud Cases", "Unknown")

    st.divider()

    # ==========================
    # Preview
    # ==========================
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.divider()

    # ==========================
    # Column Validation
    # ==========================
    expected_columns = [
        "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
        "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
        "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27",
        "V28", "Amount"
    ]

    temp = df.copy()

    if "Class" in temp.columns:
        temp = temp.drop(columns=["Class"])

    missing = [col for col in expected_columns if col not in temp.columns]

    if missing:
        st.error(f"❌ Missing Columns: {missing}")
        st.stop()

    temp = temp[expected_columns]

    st.success(" Dataset validation completed successfully.")

    st.divider()

    # ==========================
    # Prediction Section
    # ==========================
    st.subheader("🤖 Fraud Detection")

    if st.button("🚀 Predict Fraud", use_container_width=True):

        with st.spinner("Analyzing transactions..."):

            # FIX 2: Removed the fake busy-loop. The progress bar now reflects
            # actual work: run prediction, then mark complete.
            progress = st.progress(0, text="Running model...")
            prediction = model.predict(temp)
            progress.progress(50, text="Computing probabilities...")
            probability = model.predict_proba(temp)
            progress.progress(100, text="Done!")

        display_df = df.copy()
        display_df["Prediction"] = prediction
        display_df["Fraud Probability"] = (probability[:, 1] * 100).round(2)
        display_df["Status"] = display_df["Prediction"].map({
            0: "✅ Normal",
            1: "🚨 Fraud"
        })

        fraud_count = int((prediction == 1).sum())
        normal_count = int((prediction == 0).sum())

        # FIX 3: Update session_state so sidebar metrics stay in sync
        st.session_state.total_count = len(display_df)
        st.session_state.fraud_count = fraud_count
        st.session_state.normal_count = normal_count

        st.success("✅ Prediction Completed Successfully!")
        st.balloons()

        # ==========================
        # Metrics
        # ==========================
        c1, c2, c3 = st.columns(3)
        c1.metric("📄 Total Transactions", len(display_df))
        c2.metric("🚨 Fraud Transactions", fraud_count)
        c3.metric("✅ Normal Transactions", normal_count)

        st.divider()

        # FIX 4: search and all result sections are now inside the button block
        # so display_df is guaranteed to exist when they run.
        search = st.text_input("🔍 Search Transaction")

        if search:
            display_df = display_df[
                display_df.astype(str).apply(
                    lambda x: x.str.contains(search, case=False)
                ).any(axis=1)
            ]

        # ==========================
        # Prediction Table
        # ==========================
        st.subheader("📋 Prediction Results")
        st.dataframe(display_df, use_container_width=True)

        st.divider()

        # ==========================
        # Charts
        # ==========================
        chart_df = pd.DataFrame({
            "Category": ["Normal", "Fraud"],
            "Count": [normal_count, fraud_count]
        })

        left, right = st.columns(2)

        with left:
            st.subheader("📊 Bar Chart")
            st.bar_chart(chart_df.set_index("Category"))

        with right:
            st.subheader("🥧 Pie Chart")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(2, 2))
            ax.pie(
                chart_df["Count"],
                labels=chart_df["Category"],
                autopct="%1.1f%%",
                startangle=10

            )
            ax.axis("equal")
            st.pyplot(fig)

        st.divider()

        # ==========================
        # Fraud Transactions Only
        # ==========================
        fraud_df = display_df[display_df["Prediction"] == 1]

        st.subheader("🚨 Detected Fraud Transactions")

        if fraud_df.empty:
            st.success("No Fraud Transactions Detected 🎉")
        else:
            st.dataframe(fraud_df, use_container_width=True)

        st.divider()



        st.markdown(
            """
            <div class="footer">
            ❤️ Developed by <b>Prayag Sapariya</b><br>
            Credit Card Fraud Detection using Machine Learning
            </div>
            """,
            unsafe_allow_html=True
        )