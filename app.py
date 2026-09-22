import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Spam / Not Spam Detection by RNN",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# THEME CONFIGURATION
# -----------------------------------------------------------------------------
THEMES = {
    "Ocean Blue": {
        "primary": "#1E88E5",
        "bg_gradient": "linear-gradient(135deg, #0d1b2a 0%, #1b263b 100%)",
        "card_bg": "#1e293b",
        "card_border": "#334155",
        "text": "#f8fafc",
        "accent": "#38bdf8",
    },
    "Purple Galaxy": {
        "primary": "#8b5cf6",
        "bg_gradient": "linear-gradient(135deg, #180929 0%, #2e1065 100%)",
        "card_bg": "#241242",
        "card_border": "#4c1d95",
        "text": "#f5f3ff",
        "accent": "#c084fc",
    },
    "Emerald Green": {
        "primary": "#10b981",
        "bg_gradient": "linear-gradient(135deg, #062419 0%, #064e3b 100%)",
        "card_bg": "#0b382c",
        "card_border": "#047857",
        "text": "#ecfdf5",
        "accent": "#34d399",
    },
    "Sunset Orange": {
        "primary": "#f97316",
        "bg_gradient": "linear-gradient(135deg, #2b1105 0%, #431407 100%)",
        "card_bg": "#3c1a0e",
        "card_border": "#7c2d12",
        "text": "#fff7ed",
        "accent": "#fb923c",
    },
    "Midnight Dark": {
        "primary": "#6366f1",
        "bg_gradient": "linear-gradient(135deg, #09090b 0%, #18181b 100%)",
        "card_bg": "#1c1917",
        "card_border": "#27272a",
        "text": "#fafafa",
        "accent": "#a1a1aa",
    },
    "Professional Light": {
        "primary": "#2563eb",
        "bg_gradient": "linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%)",
        "card_bg": "#ffffff",
        "card_border": "#cbd5e1",
        "text": "#0f172a",
        "accent": "#1d4ed8",
    },
}

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "current_message" not in st.session_state:
    st.session_state.current_message = ""

# -----------------------------------------------------------------------------
# CACHED ARTIFACT LOADER WITH ERROR HANDLING
# -----------------------------------------------------------------------------
def resolve_file(base_candidates):
    for candidate in base_candidates:
        if os.path.exists(candidate):
            return candidate
    return None


@st.cache_resource(show_spinner="Loading RNN model and preprocessing artifacts...")
def load_all_artifacts():
    model_candidates = ["spam_rnn.keras", "spam_rnn(2).keras"]
    tokenizer_candidates = ["tokenizer.pkl", "tokenizer(4).pkl"]
    config_candidates = ["model_config.pkl", "model_config(1).pkl"]
    label_encoder_candidates = ["label_encoder.pkl", "label_encoder(1).pkl"]

    model_path = resolve_file(model_candidates)
    tok_path = resolve_file(tokenizer_candidates)
    cfg_path = resolve_file(config_candidates)
    le_path = resolve_file(label_encoder_candidates)

    missing = []
    if not model_path:
        missing.append("spam_rnn.keras")
    if not tok_path:
        missing.append("tokenizer.pkl")
    if not cfg_path:
        missing.append("model_config.pkl")
    if not le_path:
        missing.append("label_encoder.pkl")

    if missing:
        return None, None, None, None, f"Missing required files: {', '.join(missing)}"

    try:
        model = load_model(model_path, compile=False)
    except Exception as e:
        return None, None, None, None, f"Failed to load Keras model: {str(e)}"

    try:
        with open(tok_path, "rb") as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        return None, None, None, None, f"Failed to load tokenizer: {str(e)}"

    try:
        with open(cfg_path, "rb") as f:
            model_config = pickle.load(f)
    except Exception as e:
        return None, None, None, None, f"Failed to load model_config: {str(e)}"

    try:
        with open(le_path, "rb") as f:
            label_encoder = pickle.load(f)
    except Exception as e:
        return None, None, None, None, f"Failed to load label_encoder: {str(e)}"

    return model, tokenizer, model_config, label_encoder, None


model, tokenizer, model_config, label_encoder, load_error = load_all_artifacts()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ Navigation & Control")
    nav_option = st.radio(
        "Go to",
        [
            "Dashboard",
            "Message Prediction",
            "Prediction Analytics",
            "Prediction History",
            "About Model",
        ],
    )

    st.markdown("---")
    st.markdown("### 🎨 Theme Selector")
    selected_theme_name = st.selectbox(
        "Choose Dashboard Theme", list(THEMES.keys()), index=0
    )
    theme = THEMES[selected_theme_name]

    st.markdown("---")
    st.markdown("### 📊 Active Session Stats")
    total_preds = len(st.session_state.history)
    spam_preds = sum(1 for item in st.session_state.history if item["Class"] == "Spam")
    ham_preds = total_preds - spam_preds

    st.metric("Total Analyzed", total_preds)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Spam", spam_preds)
    with col_s2:
        st.metric("Not Spam", ham_preds)

    st.markdown("---")
    st.markdown("### ℹ️ Artifact Metadata")
    if model_config:
        st.caption(f"**Architecture**: {model_config.get('model_type', 'LSTM')}")
        st.caption(f"**Max Sequence**: {model_config.get('max_length', 100)}")
        st.caption(f"**Vocab Size**: {model_config.get('vocab_size', 8143)}")
        st.caption(f"**Threshold**: {model_config.get('threshold', 0.5)}")
    else:
        st.caption("Metadata unavailable")

# -----------------------------------------------------------------------------
# CUSTOM CSS INJECTION
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background: {theme['bg_gradient']};
        color: {theme['text']};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    .metric-card {{
        background-color: {theme['card_bg']};
        border: 1px solid {theme['card_border']};
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 15px;
    }}
    .status-badge-spam {{
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }}
    .status-badge-ham {{
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }}
    h1, h2, h3, h4 {{
        color: {theme['text']} !important;
    }}
    .stTextArea textarea {{
        background-color: {theme['card_bg']} !important;
        color: {theme['text']} !important;
        border: 1px solid {theme['card_border']} !important;
        border-radius: 8px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# HEADER SECTION
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align: center; padding: 25px 0 10px 0;">
        <h1 style="font-size: 2.7rem; font-weight: 800; margin-bottom: 0px;">
            Spam / Not Spam Detection by RNN
        </h1>
        <p style="font-size: 1.2rem; opacity: 0.85; margin-top: 5px;">
            Intelligent SMS and Text Classification Using Deep Learning
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if load_error:
    st.error(f"⚠️ Model Initialization Alert: {load_error}")
    st.stop()

# -----------------------------------------------------------------------------
# PREDICTION HELPER
# -----------------------------------------------------------------------------
def predict_message(raw_text):
    max_len = model_config.get("max_length", 100)
    padding_type = model_config.get("padding", "post")
    trunc_type = model_config.get("truncating", "post")
    threshold = float(model_config.get("threshold", 0.5))

    sequences = tokenizer.texts_to_sequences([raw_text])
    padded_seq = pad_sequences(
        sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type
    )

    pred_raw = model.predict(padded_seq, verbose=0)
    output_shape = pred_raw.shape

    classes = list(label_encoder.classes_)
    # classes: ['ham', 'spam'] -> 0: ham, 1: spam

    if len(output_shape) == 2 and output_shape[1] == 1:
        # Binary Sigmoid
        spam_prob = float(pred_raw[0][0])
        ham_prob = 1.0 - spam_prob
        predicted_class = "Spam" if spam_prob >= threshold else "Not Spam"
        confidence = spam_prob if predicted_class == "Spam" else ham_prob
    elif len(output_shape) == 2 and output_shape[1] == 2:
        # 2-class Softmax
        # Match ordering via label encoder
        ham_idx = classes.index("ham") if "ham" in classes else 0
        spam_idx = classes.index("spam") if "spam" in classes else 1

        ham_prob = float(pred_raw[0][ham_idx])
        spam_prob = float(pred_raw[0][spam_idx])
        predicted_class = "Spam" if spam_prob >= threshold else "Not Spam"
        confidence = max(ham_prob, spam_prob)
    else:
        raise ValueError(f"Unsupported model prediction output shape: {output_shape}")

    return {
        "class": predicted_class,
        "spam_prob": spam_prob,
        "ham_prob": ham_prob,
        "confidence": confidence,
        "threshold": threshold,
    }

# -----------------------------------------------------------------------------
# SECTION 1: DASHBOARD OVERVIEW
# -----------------------------------------------------------------------------
if nav_option == "Dashboard":
    st.markdown("### 📌 Executive Overview")

    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        st.markdown(
            f"""
            <div class="metric-card">
                <span style="font-size: 0.9rem; opacity: 0.8;">Model Core</span>
                <h2 style="margin: 5px 0;">LSTM RNN</h2>
                <span style="color: {theme['accent']}; font-size: 0.85rem;">Bidirectional Memory</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_d2:
        st.markdown(
            f"""
            <div class="metric-card">
                <span style="font-size: 0.9rem; opacity: 0.8;">Decision Cutoff</span>
                <h2 style="margin: 5px 0;">{model_config.get('threshold', 0.5):.2f}</h2>
                <span style="color: {theme['accent']}; font-size: 0.85rem;">Sigmoid Probability</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_d3:
        st.markdown(
            f"""
            <div class="metric-card">
                <span style="font-size: 0.9rem; opacity: 0.8;">Vocabulary Cap</span>
                <h2 style="margin: 5px 0;">{model_config.get('vocab_size', 8143):,}</h2>
                <span style="color: {theme['accent']}; font-size: 0.85rem;">Pre-Trained Tokens</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_d4:
        st.markdown(
            f"""
            <div class="metric-card">
                <span style="font-size: 0.9rem; opacity: 0.8;">Sequence Limit</span>
                <h2 style="margin: 5px 0;">{model_config.get('max_length', 100)} tokens</h2>
                <span style="color: {theme['accent']}; font-size: 0.85rem;">Post-Padded Input</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### ⚡ Quick Interactive Classification")

    quick_col1, quick_col2 = st.columns([1, 1])
    with quick_col1:
        sample_choice = st.selectbox(
            "Select Sample Benchmark",
            [
                "Custom Input",
                "Hey, are we still meeting tomorrow at 5 PM?",
                "Congratulations! You have won a free prize. Claim your reward now!",
            ],
        )
        default_val = "" if sample_choice == "Custom Input" else sample_choice
        quick_text = st.text_area(
            "Message Content",
            value=default_val,
            height=120,
            placeholder="Type or select a message to test...",
        )
        run_quick = st.button("Run Instant Evaluation", type="primary")

    with quick_col2:
        if run_quick and quick_text.strip():
            res = predict_message(quick_text.strip())
            badge_class = (
                "status-badge-spam"
                if res["class"] == "Spam"
                else "status-badge-ham"
            )
            st.markdown(
                f"""
                <div class="metric-card" style="text-align: center;">
                    <div style="margin-bottom: 12px;">
                        <span class="{badge_class}">{res['class'].upper()}</span>
                    </div>
                    <p style="margin: 5px 0;">Spam Probability: <b>{res['spam_prob']*100:.2f}%</b></p>
                    <p style="margin: 5px 0;">Not Spam Probability: <b>{res['ham_prob']*100:.2f}%</b></p>
                    <p style="margin: 5px 0;">Model Confidence: <b>{res['confidence']*100:.2f}%</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Record session
            st.session_state.history.append(
                {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Message": quick_text.strip(),
                    "Class": res["class"],
                    "Spam Probability (%)": round(res["spam_prob"] * 100, 2),
                    "Not Spam Probability (%)": round(res["ham_prob"] * 100, 2),
                    "Confidence (%)": round(res["confidence"] * 100, 2),
                }
            )
        elif run_quick:
            st.warning("Please type or select a valid non-empty message.")
        else:
            st.info("Input a sentence or choose an example to view real-time predictions.")

# -----------------------------------------------------------------------------
# SECTION 2: MESSAGE PREDICTION
# -----------------------------------------------------------------------------
elif nav_option == "Message Prediction":
    st.markdown("### 💬 Real-Time Message Analysis")

    sample_prompts = {
        "Clear Input": "",
        "Normal Example": "Hey, are we still meeting tomorrow at 5 PM?",
        "Spam Example": "Congratulations! You have won a free prize. Claim your reward now!",
    }

    selected_sample = st.selectbox(
        "Choose an Example Message or Start Fresh:", list(sample_prompts.keys())
    )

    if selected_sample != "Clear Input":
        user_input_area = st.text_area(
            "Enter your text / SMS message below:",
            value=sample_prompts[selected_sample],
            height=150,
            key="user_msg_input",
        )
    else:
        user_input_area = st.text_area(
            "Enter your text / SMS message below:",
            value="",
            height=150,
            key="user_msg_input_empty",
        )

    chars_count = len(user_input_area)
    words_count = len(user_input_area.split()) if user_input_area.strip() else 0

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.caption(f"**Character Count:** {chars_count}")
    with col_meta2:
        st.caption(f"**Word Count:** {words_count}")

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        analyze_clicked = st.button(
            "Analyze Message", type="primary", use_container_width=True
        )
    with col_btn2:
        clear_clicked = st.button("Clear Input", use_container_width=False)

    if clear_clicked:
        st.rerun()

    if analyze_clicked:
        if not user_input_area.strip():
            st.error("Message content cannot be blank. Please enter text to classify.")
        else:
            with st.spinner("Tokenizing, padding, and running LSTM inferences..."):
                try:
                    res = predict_message(user_input_area.strip())

                    # Record to history
                    st.session_state.history.append(
                        {
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Message": user_input_area.strip(),
                            "Class": res["class"],
                            "Spam Probability (%)": round(res["spam_prob"] * 100, 2),
                            "Not Spam Probability (%)": round(res["ham_prob"] * 100, 2),
                            "Confidence (%)": round(res["confidence"] * 100, 2),
                        }
                    )

                    st.markdown("---")
                    st.markdown("### 🎯 Classification Verdict")

                    if res["class"] == "Spam":
                        st.error(
                            f"🚨 **SPAM DETECTED** (Predicted Class: Spam | Probability: {res['spam_prob']*100:.2f}%)"
                        )
                    else:
                        st.success(
                            f"✅ **LEGITIMATE MESSAGE** (Predicted Class: Not Spam | Probability: {res['ham_prob']*100:.2f}%)"
                        )

                    col_res1, col_res2, col_res3 = st.columns(3)
                    with col_res1:
                        st.metric(
                            "Spam Probability", f"{res['spam_prob']*100:.2f}%"
                        )
                    with col_res2:
                        st.metric(
                            "Not Spam Probability", f"{res['ham_prob']*100:.2f}%"
                        )
                    with col_res3:
                        st.metric(
                            "Model Confidence", f"{res['confidence']*100:.2f}%"
                        )

                    st.progress(
                        res["confidence"],
                        text=f"Model Confidence: {res['confidence']*100:.2f}%",
                    )

                    # Charts
                    chart_col1, chart_col2 = st.columns(2)

                    with chart_col1:
                        fig_bar = go.Figure(
                            data=[
                                go.Bar(
                                    x=["Not Spam (Ham)", "Spam"],
                                    y=[res["ham_prob"] * 100, res["spam_prob"] * 100],
                                    marker_color=["#10b981", "#ef4444"],
                                    text=[
                                        f"{res['ham_prob']*100:.2f}%",
                                        f"{res['spam_prob']*100:.2f}%",
                                    ],
                                    textposition="auto",
                                )
                            ]
                        )
                        fig_bar.update_layout(
                            title="Probability Comparison",
                            yaxis_title="Probability (%)",
                            yaxis_range=[0, 100],
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color=theme["text"]),
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                    with chart_col2:
                        fig_gauge = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=res["confidence"] * 100,
                                domain={"x": [0, 1], "y": [0, 1]},
                                title={"text": "Prediction Confidence (%)"},
                                gauge={
                                    "axis": {"range": [0, 100]},
                                    "bar": {"color": theme["primary"]},
                                    "steps": [
                                        {"range": [0, 50], "color": "rgba(239, 68, 68, 0.2)"},
                                        {"range": [50, 75], "color": "rgba(245, 158, 11, 0.2)"},
                                        {"range": [75, 100], "color": "rgba(16, 185, 129, 0.2)"},
                                    ],
                                },
                            )
                        )
                        fig_gauge.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color=theme["text"]),
                        )
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    st.info(
                        f"**Interpretation**: The model evaluated the sequence against its decision threshold ({res['threshold']}). "
                        f"A threshold of {res['threshold']} classifies any output probability $\\ge {res['threshold']}$ as Spam. "
                        "Deep learning classifiers can encounter unfamiliar out-of-vocabulary phrases; predictions reflect learned recurrent sequence patterns."
                    )

                except Exception as e:
                    st.error(f"Inference error encountered: {str(e)}")

# -----------------------------------------------------------------------------
# SECTION 3: PREDICTION ANALYTICS
# -----------------------------------------------------------------------------
elif nav_option == "Prediction Analytics":
    st.markdown("### 📈 Interactive Session Analytics")

    if not st.session_state.history:
        st.info("No classification data recorded in this session yet. Run predictions to populate analytics.")
    else:
        df_history = pd.DataFrame(st.session_state.history)

        col_a1, col_a2 = st.columns(2)

        with col_a1:
            # Donut chart
            class_counts = df_history["Class"].value_counts().reset_index()
            class_counts.columns = ["Class", "Count"]

            color_map = {"Spam": "#ef4444", "Not Spam": "#10b981"}

            fig_donut = px.pie(
                class_counts,
                values="Count",
                names="Class",
                hole=0.45,
                color="Class",
                color_discrete_map=color_map,
                title="Session Prediction Distribution",
            )
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=theme["text"]),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_a2:
            # Confidence distribution over session
            fig_hist = px.histogram(
                df_history,
                x="Confidence (%)",
                color="Class",
                nbins=10,
                color_discrete_map=color_map,
                title="Confidence Spread Across Session",
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=theme["text"]),
                yaxis_title="Message Count",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Chronological trajectory
        st.markdown("#### Confidence Timeline")
        fig_line = px.line(
            df_history.reset_index(),
            x="index",
            y="Confidence (%)",
            color="Class",
            markers=True,
            hover_data=["Message"],
            color_discrete_map=color_map,
            title="Prediction Confidence Sequence",
        )
        fig_line.update_layout(
            xaxis_title="Prediction Sequence Index",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme["text"]),
        )
        st.plotly_chart(fig_line, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 4: PREDICTION HISTORY
# -----------------------------------------------------------------------------
elif nav_option == "Prediction History":
    st.markdown("### 🗂️ Active Session Log")

    if not st.session_state.history:
        st.info("The history table is currently empty. Classify text in 'Message Prediction' to view logged events.")
    else:
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, use_container_width=True)

        col_h1, col_h2 = st.columns([1, 4])
        with col_h1:
            csv_data = df_history.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export as CSV",
                data=csv_data,
                file_name=f"spam_detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_h2:
            if st.button("🗑️ Clear History Log"):
                st.session_state.history = []
                st.rerun()

# -----------------------------------------------------------------------------
# SECTION 5: ABOUT MODEL
# -----------------------------------------------------------------------------
elif nav_option == "About Model":
    st.markdown("### 🧠 Model Architecture & Deep Learning Pipeline")

    st.markdown(
        """
        #### 1. What is Spam Detection?
        Spam detection is an essential natural language processing task where unsolicited, fraudulent, or malicious 
        messages (spam) are systematically separated from genuine communications (ham). Unlike rigid rule-based systems, 
        neural text classifiers learn semantic dependencies, punctuation patterns, and contextual relationships.

        #### 2. Recurrent Neural Networks (RNN) & LSTM
        * **Recurrent Neural Networks (RNNs)** process sequential data by passing hidden states across sequence steps. 
          However, standard RNNs often struggle with the vanishing gradient problem over long sequences.
        * **Long Short-Term Memory (LSTM)** networks solve this by employing a memory cell regulated by three gating mechanisms:
          - **Forget Gate**: Decides what proportion of past cell state information to discard.
          - **Input Gate**: Selects which new candidate features to update into the cell state.
          - **Output Gate**: Controls what elements of the current state are emitted as the output hidden vector.

        #### 3. Preprocessing & Prediction Pipeline
        The inference process converts raw text into numerical predictions strictly using the serialized assets:
        """
    )

    st.markdown(
        f"""
        ```
        Raw Message 
             │
             ▼
        Tokenization (tokenizer.pkl, vocab_size = {model_config.get('vocab_size', 8143)})
             │
             ▼
        Sequence Padding (maxlen = {model_config.get('max_length', 100)}, padding = '{model_config.get('padding', 'post')}')
             │
             ▼
        Embedding Layer (Embedding dim = 64, mask_zero = True)
             │
             ▼
        LSTM Layer (64 memory units, dropout = 0.3)
             │
             ▼
        Dense Projection (32 units, ReLU activation, dropout = 0.3)
             │
             ▼
        Output Layer (1 unit, Sigmoid activation) ──► Decision Threshold ({model_config.get('threshold', 0.5)}) ──► Spam / Not Spam
        ```
        """
    )

    st.markdown("#### 4. Verified Model Configuration")
    config_details = {
        "Configuration Key": list(model_config.keys()),
        "Saved Value": [str(v) for v in model_config.values()],
    }
    st.table(pd.DataFrame(config_details))

    st.markdown("#### 5. Label Encoder Classes")
    le_df = pd.DataFrame(
        {
            "Class Index": list(range(len(label_encoder.classes_))),
            "Encoded Label": list(label_encoder.classes_),
            "Semantic Interpretation": [
                "Legitimate Message (Not Spam)" if c == "ham" else "Spam Message"
                for c in label_encoder.classes_
            ],
        }
    )
    st.table(le_df)
