import os
import time
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Limit multi-threading overhead on containerized platforms like Render
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], "GPU")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "spam-rnn-detection-secret-key")

# Base directory for resolving relative model assets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "spam_rnn.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
CONFIG_PATH = os.path.join(BASE_DIR, "model_config.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

# Alternative filename fallback if files were uploaded with index suffixes
if not os.path.exists(CONFIG_PATH):
    alt_config = os.path.join(BASE_DIR, "model_config(1).pkl")
    if os.path.exists(alt_config):
        CONFIG_PATH = alt_config

MODEL = None
TOKENIZER = None
CONFIG = None
LABEL_ENCODER = None
INIT_ERROR = None


def load_artifacts():
    """Load model, tokenizer, config, and label encoder once during startup."""
    global MODEL, TOKENIZER, CONFIG, LABEL_ENCODER, INIT_ERROR
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file missing at: {MODEL_PATH}")
        if not os.path.exists(TOKENIZER_PATH):
            raise FileNotFoundError(f"Tokenizer missing at: {TOKENIZER_PATH}")
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Config file missing at: {CONFIG_PATH}")
        if not os.path.exists(LABEL_ENCODER_PATH):
            raise FileNotFoundError(f"Label encoder missing at: {LABEL_ENCODER_PATH}")

        with open(CONFIG_PATH, "rb") as f:
            CONFIG = pickle.load(f)

        with open(TOKENIZER_PATH, "rb") as f:
            TOKENIZER = pickle.load(f)

        with open(LABEL_ENCODER_PATH, "rb") as f:
            LABEL_ENCODER = pickle.load(f)

        # Load Keras LSTM model
        MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)

        # Warm-up predict run to prime graph/execution trace
        max_len = CONFIG.get("max_length", 100)
        dummy_seq = np.zeros((1, max_len), dtype=np.float32)
        _ = MODEL.predict(dummy_seq, verbose=0)
        print("All models and preprocessing artifacts successfully loaded and warmed up.")
    except Exception as e:
        INIT_ERROR = str(e)
        print(f"Error loading model artifacts: {e}")


# Initialize artifacts on worker startup
load_artifacts()


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="cyberpunk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spam Not Spam Detection by RNN | AI Dashboard</title>
    <!-- Modern Font and Icons -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        :root {
            --transition-speed: 0.3s;
        }

        /* Themes */
        html[data-theme="cyberpunk"] {
            --bg-primary: #07090e;
            --bg-secondary: #0f1422;
            --surface: rgba(18, 24, 38, 0.75);
            --surface-border: rgba(99, 102, 241, 0.25);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.4);
            --neon-blue: #06b6d4;
            --neon-purple: #a855f7;
            --spam-color: #f43f5e;
            --spam-bg: rgba(244, 63, 94, 0.12);
            --ham-color: #10b981;
            --ham-bg: rgba(16, 185, 129, 0.12);
            --card-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.7);
        }

        html[data-theme="ocean"] {
            --bg-primary: #081226;
            --bg-secondary: #0d1e3d;
            --surface: rgba(13, 30, 61, 0.75);
            --surface-border: rgba(14, 165, 233, 0.25);
            --text-primary: #f0f9ff;
            --text-secondary: #93c5fd;
            --text-muted: #60a5fa;
            --accent: #0284c7;
            --accent-glow: rgba(2, 132, 199, 0.4);
            --neon-blue: #38bdf8;
            --neon-purple: #2563eb;
            --spam-color: #fb7185;
            --spam-bg: rgba(251, 113, 133, 0.12);
            --ham-color: #34d399;
            --ham-bg: rgba(52, 211, 153, 0.12);
            --card-shadow: 0 10px 30px -10px rgba(2, 132, 199, 0.3);
        }

        html[data-theme="emerald"] {
            --bg-primary: #051a14;
            --bg-secondary: #0a2920;
            --surface: rgba(10, 41, 32, 0.78);
            --surface-border: rgba(16, 185, 129, 0.28);
            --text-primary: #ecfdf5;
            --text-secondary: #a7f3d0;
            --text-muted: #6ee7b7;
            --accent: #10b981;
            --accent-glow: rgba(16, 185, 129, 0.4);
            --neon-blue: #2dd4bf;
            --neon-purple: #059669;
            --spam-color: #f87171;
            --spam-bg: rgba(248, 113, 113, 0.14);
            --ham-color: #10b981;
            --ham-bg: rgba(16, 185, 129, 0.15);
            --card-shadow: 0 10px 30px -10px rgba(16, 185, 129, 0.3);
        }

        html[data-theme="sunset"] {
            --bg-primary: #180c1e;
            --bg-secondary: #261230;
            --surface: rgba(43, 20, 54, 0.78);
            --surface-border: rgba(244, 63, 94, 0.3);
            --text-primary: #fff1f2;
            --text-secondary: #fda4af;
            --text-muted: #fb7185;
            --accent: #f43f5e;
            --accent-glow: rgba(244, 63, 94, 0.4);
            --neon-blue: #fb923c;
            --neon-purple: #c084fc;
            --spam-color: #ef4444;
            --spam-bg: rgba(239, 68, 68, 0.15);
            --ham-color: #22c55e;
            --ham-bg: rgba(34, 197, 94, 0.15);
            --card-shadow: 0 10px 30px -10px rgba(244, 63, 94, 0.3);
        }

        html[data-theme="light"] {
            --bg-primary: #f8fafc;
            --bg-secondary: #f1f5f9;
            --surface: rgba(255, 255, 255, 0.9);
            --surface-border: rgba(203, 213, 225, 0.8);
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #94a3b8;
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.25);
            --neon-blue: #0284c7;
            --neon-purple: #7c3aed;
            --spam-color: #dc2626;
            --spam-bg: rgba(220, 38, 38, 0.1);
            --ham-color: #059669;
            --ham-bg: rgba(5, 150, 105, 0.1);
            --card-shadow: 0 8px 24px -6px rgba(15, 23, 42, 0.08);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
            transition: background-color var(--transition-speed), border-color var(--transition-speed);
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        /* Glassmorphism Card Style */
        .glass-card {
            background: var(--surface);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid var(--surface-border);
            border-radius: 18px;
            box-shadow: var(--card-shadow);
            padding: 24px;
        }

        .container {
            max-width: 1380px;
            margin: 0 auto;
            padding: 24px 20px;
            width: 100%;
        }

        /* Header Navigation */
        header {
            border-bottom: 1px solid var(--surface-border);
            background: var(--surface);
            backdrop-filter: blur(16px);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .nav-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1380px;
            margin: 0 auto;
            padding: 16px 20px;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .brand-logo {
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, var(--neon-blue), var(--neon-purple));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 20px;
            box-shadow: 0 4px 15px var(--accent-glow);
            animation: pulse-slow 3s infinite alternate;
        }

        @keyframes pulse-slow {
            0% { transform: scale(1); filter: brightness(1); }
            100% { transform: scale(1.05); filter: brightness(1.2); }
        }

        .brand-text h1 {
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(90deg, var(--text-primary), var(--text-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-text p {
            font-size: 0.78rem;
            color: var(--text-muted);
            font-weight: 500;
        }

        .nav-controls {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .theme-selector {
            display: flex;
            align-items: center;
            gap: 8px;
            background: var(--bg-secondary);
            padding: 6px 12px;
            border-radius: 12px;
            border: 1px solid var(--surface-border);
            font-size: 0.85rem;
        }

        .theme-selector select {
            background: transparent;
            border: none;
            color: var(--text-primary);
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            outline: none;
        }

        .theme-selector select option {
            background-color: var(--bg-primary);
            color: var(--text-primary);
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            border: 1px solid;
        }

        .status-badge.online {
            color: var(--ham-color);
            background: var(--ham-bg);
            border-color: rgba(16, 185, 129, 0.3);
        }

        .status-badge.offline {
            color: var(--spam-color);
            background: var(--spam-bg);
            border-color: rgba(244, 63, 94, 0.3);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: currentColor;
            box-shadow: 0 0 8px currentColor;
        }

        /* Layout Grid */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 24px;
            margin-top: 10px;
        }

        @media (max-width: 980px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Section Headings */
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
        }

        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .section-title i {
            color: var(--neon-blue);
        }

        /* Message Input Workspace */
        .textarea-wrapper {
            position: relative;
            margin-bottom: 14px;
        }

        textarea {
            width: 100%;
            height: 180px;
            background: var(--bg-secondary);
            border: 1.5px solid var(--surface-border);
            border-radius: 14px;
            padding: 16px;
            color: var(--text-primary);
            font-size: 0.95rem;
            line-height: 1.5;
            resize: vertical;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        textarea:focus {
            border-color: var(--neon-blue);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        .textarea-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 8px;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Button Cluster */
        .action-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 14px;
            align-items: center;
        }

        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 10px 18px;
            border-radius: 10px;
            font-size: 0.88rem;
            font-weight: 600;
            cursor: pointer;
            border: none;
            outline: none;
            transition: all 0.2s ease;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--neon-blue), var(--accent));
            color: white;
            box-shadow: 0 4px 14px var(--accent-glow);
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            filter: brightness(1.1);
            box-shadow: 0 6px 20px var(--accent-glow);
        }

        .btn-primary:active {
            transform: translateY(0);
        }

        .btn-secondary {
            background: var(--bg-secondary);
            color: var(--text-secondary);
            border: 1px solid var(--surface-border);
        }

        .btn-secondary:hover {
            background: var(--surface);
            color: var(--text-primary);
            border-color: var(--neon-blue);
        }

        .btn-sample {
            font-size: 0.78rem;
            padding: 6px 12px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--surface-border);
            color: var(--text-muted);
            border-radius: 8px;
        }

        .btn-sample:hover {
            color: var(--text-primary);
            border-color: var(--neon-purple);
        }

        /* Prediction Result Section */
        .result-container {
            margin-top: 24px;
            display: none;
        }

        .result-card {
            border-radius: 16px;
            padding: 22px;
            border: 2px solid;
            position: relative;
            overflow: hidden;
            animation: fadeIn 0.4s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-card.spam {
            background: var(--spam-bg);
            border-color: var(--spam-color);
        }

        .result-card.ham {
            background: var(--ham-bg);
            border-color: var(--ham-color);
        }

        .result-headline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }

        .badge-status {
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .result-card.spam .badge-status { color: var(--spam-color); }
        .result-card.ham .badge-status { color: var(--ham-color); }

        .confidence-label {
            font-size: 1.6rem;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Progress Bar */
        .progress-bar-bg {
            height: 10px;
            background: rgba(0, 0, 0, 0.25);
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 18px;
        }

        .progress-bar-fill {
            height: 100%;
            border-radius: 8px;
            transition: width 1s ease-in-out;
        }

        .result-card.spam .progress-bar-fill { background: var(--spam-color); }
        .result-card.ham .progress-bar-fill { background: var(--ham-color); }

        .prob-metrics {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 12px;
            font-size: 0.82rem;
            font-family: 'JetBrains Mono', monospace;
        }

        .metric-box {
            background: rgba(0, 0, 0, 0.15);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }

        .metric-title {
            color: var(--text-muted);
            margin-bottom: 4px;
            font-size: 0.72rem;
            text-transform: uppercase;
        }

        .metric-value {
            font-weight: 700;
            font-size: 0.95rem;
            color: var(--text-primary);
        }

        /* Analytics & Stat Cards */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }

        .stat-card {
            background: var(--bg-secondary);
            border: 1px solid var(--surface-border);
            padding: 14px;
            border-radius: 12px;
            text-align: center;
        }

        .stat-card .val {
            font-size: 1.35rem;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            color: var(--neon-blue);
            margin-top: 4px;
        }

        .stat-card .lbl {
            font-size: 0.72rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
        }

        /* Charts Layout */
        .chart-box {
            position: relative;
            margin-bottom: 20px;
            background: var(--bg-secondary);
            border: 1px solid var(--surface-border);
            padding: 16px;
            border-radius: 14px;
        }

        /* History Table */
        .history-card {
            margin-top: 24px;
        }

        .history-table-container {
            max-height: 280px;
            overflow-y: auto;
            border: 1px solid var(--surface-border);
            border-radius: 12px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            text-align: left;
        }

        th {
            background: var(--bg-secondary);
            padding: 12px 14px;
            font-weight: 700;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--surface-border);
            position: sticky;
            top: 0;
            z-index: 1;
        }

        td {
            padding: 12px 14px;
            border-bottom: 1px solid var(--surface-border);
            color: var(--text-secondary);
        }

        tr:hover td {
            background: rgba(255, 255, 255, 0.02);
            color: var(--text-primary);
        }

        .msg-snippet {
            max-width: 320px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
        }

        .tag {
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: inline-block;
        }

        .tag.spam {
            background: var(--spam-bg);
            color: var(--spam-color);
            border: 1px solid var(--spam-color);
        }

        .tag.ham {
            background: var(--ham-bg);
            color: var(--ham-color);
            border: 1px solid var(--ham-color);
        }

        /* Toast Notifications */
        #toast {
            position: fixed;
            bottom: 24px;
            right: 24px;
            padding: 14px 20px;
            border-radius: 12px;
            background: var(--surface);
            color: var(--text-primary);
            border: 1px solid var(--surface-border);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 0.88rem;
            font-weight: 600;
            transform: translateY(100px);
            opacity: 0;
            transition: all 0.3s cubic-bezier(0.68, -0.55, 0.27, 1.55);
            z-index: 1000;
        }

        #toast.show {
            transform: translateY(0);
            opacity: 1;
        }

        /* Loading Overlay Spinner */
        .spinner {
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Footer */
        footer {
            margin-top: auto;
            border-top: 1px solid var(--surface-border);
            padding: 20px;
            text-align: center;
            font-size: 0.82rem;
            color: var(--text-muted);
            background: var(--surface);
        }

        .footer-content {
            max-width: 1380px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }

        .model-pills {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .pill {
            padding: 3px 10px;
            background: var(--bg-secondary);
            border-radius: 6px;
            font-size: 0.72rem;
            font-family: 'JetBrains Mono', monospace;
            border: 1px solid var(--surface-border);
            color: var(--text-secondary);
        }
    </style>
</head>
<body>

    <!-- Header Navigation -->
    <header>
        <div class="nav-content">
            <div class="brand">
                <div class="brand-logo">
                    <i class="fa-solid fa-brain"></i>
                </div>
                <div class="brand-text">
                    <h1>Spam Not Spam Detection by RNN</h1>
                    <p>Intelligent SMS and Message Classification Using Deep Learning</p>
                </div>
            </div>
            <div class="nav-controls">
                <div class="theme-selector">
                    <i class="fa-solid fa-palette" style="color: var(--neon-blue);"></i>
                    <select id="themeSelect" onchange="changeTheme(this.value)">
                        <option value="cyberpunk">Dark Cyberpunk</option>
                        <option value="ocean">Ocean Blue</option>
                        <option value="emerald">Emerald Green</option>
                        <option value="sunset">Sunset Gradient</option>
                        <option value="light">Professional Light</option>
                    </select>
                </div>
                <div class="status-badge {{ 'online' if not init_error else 'offline' }}">
                    <div class="status-dot"></div>
                    <span>{{ 'LSTM Active' if not init_error else 'Model Error' }}</span>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Workspace -->
    <main class="container">
        <div class="dashboard-grid">
            
            <!-- Left Column: Input & Results -->
            <div class="left-col">
                <div class="glass-card">
                    <div class="section-header">
                        <span class="section-title">
                            <i class="fa-solid fa-message"></i> Message Prediction Studio
                        </span>
                        <div style="display: flex; gap: 8px;">
                            <button class="btn btn-sample" onclick="loadSample('spam')">
                                <i class="fa-solid fa-triangle-exclamation"></i> Spam Sample
                            </button>
                            <button class="btn btn-sample" onclick="loadSample('ham')">
                                <i class="fa-solid fa-shield-halved"></i> Safe Sample
                            </button>
                        </div>
                    </div>

                    <div class="textarea-wrapper">
                        <textarea id="messageInput" placeholder="Paste or type SMS, message, or email text here to classify..." oninput="updateCounters()"></textarea>
                        <div class="textarea-meta">
                            <span><i class="fa-regular fa-keyboard"></i> <span id="charCount">0</span> characters | <span id="wordCount">0</span> words</span>
                            <span>Max Sequence Length: <strong>{{ max_length }}</strong></span>
                        </div>
                    </div>

                    <div class="action-row">
                        <button class="btn btn-primary" id="predictBtn" onclick="predictMessage()">
                            <i class="fa-solid fa-bolt"></i> <span>Analyze Message</span>
                        </button>
                        <button class="btn btn-secondary" onclick="clearMessage()">
                            <i class="fa-solid fa-eraser"></i> Clear
                        </button>
                        <button class="btn btn-secondary" onclick="copyResult()" id="copyBtn" style="display: none;">
                            <i class="fa-regular fa-copy"></i> Copy Result
                        </button>
                    </div>

                    <!-- Prediction Result Card -->
                    <div class="result-container" id="resultContainer">
                        <div class="result-card" id="resultCard">
                            <div class="result-headline">
                                <div class="badge-status" id="resultBadge">
                                    <i id="resultIcon" class="fa-solid"></i>
                                    <span id="resultLabel">RESULT</span>
                                </div>
                                <div class="confidence-label" id="confidenceValue">0.0%</div>
                            </div>
                            
                            <div class="progress-bar-bg">
                                <div class="progress-bar-fill" id="progressBar" style="width: 0%;"></div>
                            </div>

                            <div class="prob-metrics">
                                <div class="metric-box">
                                    <div class="metric-title">Spam Probability</div>
                                    <div class="metric-value" id="spamProb">0.0%</div>
                                </div>
                                <div class="metric-box">
                                    <div class="metric-title">Not Spam Probability</div>
                                    <div class="metric-value" id="hamProb">0.0%</div>
                                </div>
                                <div class="metric-box">
                                    <div class="metric-title">Inference Time</div>
                                    <div class="metric-value" id="inferenceTime">0 ms</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- History Table Card -->
                <div class="glass-card history-card">
                    <div class="section-header">
                        <span class="section-title">
                            <i class="fa-solid fa-clock-rotate-left"></i> Session Classification History
                        </span>
                        <button class="btn btn-secondary" style="font-size: 0.78rem; padding: 6px 12px;" onclick="clearHistory()">
                            <i class="fa-solid fa-trash-can"></i> Clear History
                        </button>
                    </div>
                    <div class="history-table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>Timestamp</th>
                                    <th>Message Snippet</th>
                                    <th>Class</th>
                                    <th>Confidence</th>
                                    <th>Spam Prob</th>
                                </tr>
                            </thead>
                            <tbody id="historyTableBody">
                                <tr id="emptyHistoryRow">
                                    <td colspan="5" style="text-align: center; color: var(--text-muted); padding: 24px;">
                                        No messages analyzed yet. Predict a message above to populate session history.
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- Right Column: Analytics & Architecture Insights -->
            <div class="right-col">
                <div class="glass-card">
                    <div class="section-header">
                        <span class="section-title">
                            <i class="fa-solid fa-chart-pie"></i> Prediction Analytics
                        </span>
                        <span style="font-size: 0.75rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;">Live Metrics</span>
                    </div>

                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="lbl">Total Checked</div>
                            <div class="val" id="statTotal">0</div>
                        </div>
                        <div class="stat-card">
                            <div class="lbl">Spam Count</div>
                            <div class="val" id="statSpam" style="color: var(--spam-color);">0</div>
                        </div>
                        <div class="stat-card">
                            <div class="lbl">Avg Confidence</div>
                            <div class="val" id="statAvgConf">0%</div>
                        </div>
                    </div>

                    <!-- Probability Chart -->
                    <div class="chart-box">
                        <canvas id="probChart" height="180"></canvas>
                    </div>

                    <!-- History Timeline Chart -->
                    <div class="chart-box" style="margin-bottom: 0;">
                        <canvas id="timelineChart" height="150"></canvas>
                    </div>
                </div>
            </div>

        </div>
    </main>

    <!-- Toast Component -->
    <div id="toast">
        <i class="fa-solid fa-circle-info" id="toastIcon"></i>
        <span id="toastMsg">Notification message</span>
    </div>

    <!-- Footer -->
    <footer>
        <div class="footer-content">
            <div>
                <strong>Spam Not Spam Detection by RNN</strong> &bull; Deep Learning Pipeline Deployment
            </div>
            <div class="model-pills">
                <span class="pill">Architecture: {{ model_type }}</span>
                <span class="pill">Max Words: {{ max_words }}</span>
                <span class="pill">Vocab Size: {{ vocab_size }}</span>
                <span class="pill">Threshold: {{ threshold }}</span>
                <span class="pill">Mapping: {{ class_mapping }}</span>
            </div>
        </div>
    </footer>

    <!-- Frontend Interactive Logic & Charts Initialization -->
    <script>
        let lastResultText = "";
        let probChart = null;
        let timelineChart = null;

        const SAMPLES = {
            spam: "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
            ham: "Hey, are we still meeting up for coffee this afternoon? Let me know when you get off work."
        };

        // Theme management
        function initTheme() {
            const savedTheme = localStorage.getItem("rnn_spam_theme") || "cyberpunk";
            document.documentElement.setAttribute("data-theme", savedTheme);
            document.getElementById("themeSelect").value = savedTheme;
        }

        function changeTheme(theme) {
            document.documentElement.setAttribute("data-theme", theme);
            localStorage.setItem("rnn_spam_theme", theme);
            updateChartTheme();
            showToast("Theme updated to " + theme, "info");
        }

        function showToast(message, type = "info") {
            const toast = document.getElementById("toast");
            const icon = document.getElementById("toastIcon");
            const msg = document.getElementById("toastMsg");

            msg.textContent = message;
            if (type === "error") {
                icon.className = "fa-solid fa-circle-exclamation";
                toast.style.borderColor = "var(--spam-color)";
            } else if (type === "success") {
                icon.className = "fa-solid fa-circle-check";
                toast.style.borderColor = "var(--ham-color)";
            } else {
                icon.className = "fa-solid fa-circle-info";
                toast.style.borderColor = "var(--surface-border)";
            }

            toast.classList.add("show");
            setTimeout(() => {
                toast.classList.remove("show");
            }, 3000);
        }

        function updateCounters() {
            const text = document.getElementById("messageInput").value;
            document.getElementById("charCount").textContent = text.length;
            const words = text.trim().length === 0 ? 0 : text.trim().split(/\\s+/).length;
            document.getElementById("wordCount").textContent = words;
        }

        function loadSample(type) {
            document.getElementById("messageInput").value = SAMPLES[type];
            updateCounters();
            showToast(`Loaded ${type.toUpperCase()} sample text`, "info");
        }

        function clearMessage() {
            document.getElementById("messageInput").value = "";
            updateCounters();
            document.getElementById("resultContainer").style.display = "none";
            document.getElementById("copyBtn").style.display = "none";
            resetCurrentGauge();
        }

        // Send predict request to Flask backend
        async function predictMessage() {
            const textarea = document.getElementById("messageInput");
            const text = textarea.value.trim();

            if (!text) {
                showToast("Please enter a valid message before analyzing.", "error");
                textarea.focus();
                return;
            }

            const predictBtn = document.getElementById("predictBtn");
            const originalBtnHtml = predictBtn.innerHTML;
            predictBtn.innerHTML = '<span class="spinner"></span> <span>Analyzing...</span>';
            predictBtn.disabled = true;

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message: text })
                });

                const data = await response.json();

                if (!response.ok || !data.success) {
                    throw new Error(data.error || "Inference server error");
                }

                displayPredictionResult(data);
                appendHistory(data, text);
                updateAnalytics();
                showToast("Prediction completed successfully!", "success");
            } catch (err) {
                showToast(err.message || "Failed to contact prediction backend", "error");
            } finally {
                predictBtn.innerHTML = originalBtnHtml;
                predictBtn.disabled = false;
            }
        }

        function displayPredictionResult(res) {
            const container = document.getElementById("resultContainer");
            const card = document.getElementById("resultCard");
            const badge = document.getElementById("resultBadge");
            const icon = document.getElementById("resultIcon");
            const label = document.getElementById("resultLabel");
            const confidenceValue = document.getElementById("confidenceValue");
            const progressBar = document.getElementById("progressBar");
            const spamProb = document.getElementById("spamProb");
            const hamProb = document.getElementById("hamProb");
            const inferenceTime = document.getElementById("inferenceTime");
            const copyBtn = document.getElementById("copyBtn");

            const isSpam = res.is_spam;
            card.className = "result-card " + (isSpam ? "spam" : "ham");
            icon.className = isSpam ? "fa-solid fa-triangle-exclamation" : "fa-solid fa-circle-check";
            label.textContent = isSpam ? "SPAM DETECTED" : "NOT SPAM – SAFE MESSAGE";
            
            confidenceValue.textContent = (res.confidence * 100).toFixed(2) + "%";
            progressBar.style.width = (res.confidence * 100).toFixed(2) + "%";

            spamProb.textContent = (res.spam_probability * 100).toFixed(2) + "%";
            hamProb.textContent = (res.ham_probability * 100).toFixed(2) + "%";
            inferenceTime.textContent = res.inference_time_ms + " ms";

            container.style.display = "block";
            copyBtn.style.display = "inline-flex";

            lastResultText = `Classification: ${label.textContent} | Confidence: ${(res.confidence * 100).toFixed(2)}% | Spam Prob: ${(res.spam_probability * 100).toFixed(2)}%`;

            // Update Doughnut Chart with the current prediction probabilities
            updateProbChart(res.spam_probability, res.ham_probability);
        }

        function copyResult() {
            if (!lastResultText) return;
            navigator.clipboard.writeText(lastResultText).then(() => {
                showToast("Result copied to clipboard!", "success");
            }).catch(() => {
                showToast("Failed to copy result.", "error");
            });
        }

        // History Management via localStorage
        function getHistory() {
            try {
                return JSON.parse(localStorage.getItem("rnn_spam_history")) || [];
            } catch {
                return [];
            }
        }

        function saveHistory(list) {
            localStorage.setItem("rnn_spam_history", JSON.stringify(list));
        }

        function appendHistory(res, originalText) {
            const list = getHistory();
            const record = {
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
                snippet: originalText.length > 60 ? originalText.substring(0, 60) + "..." : originalText,
                is_spam: res.is_spam,
                predicted_label: res.predicted_label,
                confidence: res.confidence,
                spam_probability: res.spam_probability
            };
            list.unshift(record);
            if (list.length > 50) list.pop(); // Keep last 50 predictions
            saveHistory(list);
            renderHistoryTable();
        }

        function renderHistoryTable() {
            const list = getHistory();
            const tbody = document.getElementById("historyTableBody");
            tbody.innerHTML = "";

            if (list.length === 0) {
                tbody.innerHTML = `
                    <tr id="emptyHistoryRow">
                        <td colspan="5" style="text-align: center; color: var(--text-muted); padding: 24px;">
                            No messages analyzed yet. Predict a message above to populate session history.
                        </td>
                    </tr>
                `;
                return;
            }

            list.forEach(item => {
                const tr = document.createElement("tr");
                const tagClass = item.is_spam ? "spam" : "ham";
                const labelText = item.is_spam ? "SPAM" : "HAM / SAFE";
                tr.innerHTML = `
                    <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;">${item.timestamp}</td>
                    <td class="msg-snippet" title="${item.snippet}">${item.snippet}</td>
                    <td><span class="tag ${tagClass}">${labelText}</span></td>
                    <td style="font-family: 'JetBrains Mono', monospace;">${(item.confidence * 100).toFixed(1)}%</td>
                    <td style="font-family: 'JetBrains Mono', monospace;">${(item.spam_probability * 100).toFixed(1)}%</td>
                `;
                tbody.appendChild(tr);
            });
        }

        function clearHistory() {
            localStorage.removeItem("rnn_spam_history");
            renderHistoryTable();
            updateAnalytics();
            resetCurrentGauge();
            showToast("Session history cleared.", "info");
        }

        function updateAnalytics() {
            const list = getHistory();
            const total = list.length;
            const spamCount = list.filter(i => i.is_spam).length;
            const avgConf = total > 0 ? (list.reduce((acc, curr) => acc + curr.confidence, 0) / total * 100).toFixed(1) : 0;

            document.getElementById("statTotal").textContent = total;
            document.getElementById("statSpam").textContent = spamCount;
            document.getElementById("statAvgConf").textContent = avgConf + "%";

            updateTimelineChart(list);
        }

        // Charts Setup
        function initCharts() {
            const computedStyle = getComputedStyle(document.documentElement);
            const spamColor = computedStyle.getPropertyValue('--spam-color').trim() || '#f43f5e';
            const hamColor = computedStyle.getPropertyValue('--ham-color').trim() || '#10b981';

            // 1. Current Probabilities Doughnut
            const ctxProb = document.getElementById("probChart").getContext("2d");
            probChart = new Chart(ctxProb, {
                type: 'doughnut',
                data: {
                    labels: ['Spam Probability', 'Safe Probability'],
                    datasets: [{
                        data: [50, 50],
                        backgroundColor: [spamColor, hamColor],
                        borderWidth: 0,
                        hoverOffset: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom', labels: { color: computedStyle.getPropertyValue('--text-secondary').trim(), font: { family: 'Plus Jakarta Sans', size: 11 } } },
                        title: { display: true, text: 'Latest Prediction Breakdown', color: computedStyle.getPropertyValue('--text-primary').trim(), font: { size: 12, weight: 'bold' } }
                    },
                    cutout: '70%'
                }
            });

            // 2. Timeline Chart
            const ctxTimeline = document.getElementById("timelineChart").getContext("2d");
            timelineChart = new Chart(ctxTimeline, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Spam Probability Trend (%)',
                        data: [],
                        borderColor: computedStyle.getPropertyValue('--neon-blue').trim() || '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.1)',
                        fill: true,
                        tension: 0.35,
                        pointRadius: 3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { min: 0, max: 100, ticks: { color: computedStyle.getPropertyValue('--text-muted').trim() }, grid: { color: 'rgba(255,255,255,0.05)' } },
                        x: { ticks: { color: computedStyle.getPropertyValue('--text-muted').trim() }, grid: { display: false } }
                    },
                    plugins: {
                        legend: { display: false },
                        title: { display: true, text: 'Prediction History Trend (% Spam)', color: computedStyle.getPropertyValue('--text-primary').trim(), font: { size: 12, weight: 'bold' } }
                    }
                }
            });
        }

        function updateProbChart(spamP, hamP) {
            if (!probChart) return;
            probChart.data.datasets[0].data = [
                parseFloat((spamP * 100).toFixed(2)),
                parseFloat((hamP * 100).toFixed(2))
            ];
            probChart.update();
        }

        function resetCurrentGauge() {
            if (!probChart) return;
            probChart.data.datasets[0].data = [50, 50];
            probChart.update();
        }

        function updateTimelineChart(list) {
            if (!timelineChart) return;
            const chronological = [...list].reverse().slice(-10); // Display last 10 points
            timelineChart.data.labels = chronological.map(i => i.timestamp);
            timelineChart.data.datasets[0].data = chronological.map(i => (i.spam_probability * 100).toFixed(1));
            timelineChart.update();
        }

        function updateChartTheme() {
            if (!probChart || !timelineChart) return;
            const computedStyle = getComputedStyle(document.documentElement);
            const spamColor = computedStyle.getPropertyValue('--spam-color').trim();
            const hamColor = computedStyle.getPropertyValue('--ham-color').trim();
            const textColor = computedStyle.getPropertyValue('--text-secondary').trim();

            probChart.data.datasets[0].backgroundColor = [spamColor, hamColor];
            probChart.options.plugins.legend.labels.color = textColor;
            probChart.update();

            timelineChart.data.datasets[0].borderColor = computedStyle.getPropertyValue('--neon-blue').trim();
            timelineChart.update();
        }

        // Initialize application on DOM load
        window.addEventListener("DOMContentLoaded", () => {
            initTheme();
            initCharts();
            renderHistoryTable();
            updateAnalytics();
        });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    """Render the main single-page AI dashboard."""
    max_length = CONFIG.get("max_length", 100) if CONFIG else 100
    max_words = CONFIG.get("max_words", 10000) if CONFIG else 10000
    vocab_size = CONFIG.get("vocab_size", 8143) if CONFIG else 8143
    threshold = CONFIG.get("threshold", 0.5) if CONFIG else 0.5
    model_type = CONFIG.get("model_type", "LSTM") if CONFIG else "LSTM"

    class_mapping = "Unknown"
    if LABEL_ENCODER and hasattr(LABEL_ENCODER, "classes_"):
        class_mapping = f"0={LABEL_ENCODER.classes_[0]}, 1={LABEL_ENCODER.classes_[1]}"

    return render_template_string(
        HTML_TEMPLATE,
        init_error=INIT_ERROR,
        max_length=max_length,
        max_words=max_words,
        vocab_size=vocab_size,
        threshold=threshold,
        model_type=model_type,
        class_mapping=class_mapping,
    )


@app.route("/health", methods=["GET"])
def health():
    """Service health check endpoint."""
    if INIT_ERROR:
        return jsonify({
            "status": "unhealthy",
            "error": "Model initialization failed.",
            "details": INIT_ERROR
        }), 503

    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "tokenizer_loaded": TOKENIZER is not None,
        "label_encoder_loaded": LABEL_ENCODER is not None,
        "classes": list(LABEL_ENCODER.classes_) if LABEL_ENCODER else [],
        "config": {
            "max_length": CONFIG.get("max_length", 100) if CONFIG else None,
            "threshold": CONFIG.get("threshold", 0.5) if CONFIG else None,
            "model_type": CONFIG.get("model_type", "LSTM") if CONFIG else None
        }
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Message inference endpoint.
    Pipeline: User Message -> texts_to_sequences -> pad_sequences ->
              LSTM Model Prediction -> Threshold Classification ->
              Label Encoder Mapping -> JSON Response
    """
    if INIT_ERROR or MODEL is None or TOKENIZER is None or CONFIG is None or LABEL_ENCODER is None:
        return jsonify({
            "success": False,
            "error": "Model service unavailable. Assets failed to load during startup."
        }), 503

    try:
        data = request.get_json(force=True, silent=True)
        if not data or "message" not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'message' field in JSON request."
            }), 400

        raw_message = data.get("message", "")
        if not isinstance(raw_message, str) or not raw_message.strip():
            return jsonify({
                "success": False,
                "error": "Message text cannot be empty or solely whitespace."
            }), 400

        clean_text = raw_message.strip()

        start_time = time.perf_counter()

        # Step 1: Text Tokenization and sequence conversion
        sequences = TOKENIZER.texts_to_sequences([clean_text])

        # Step 2: Sequence padding using fitted configuration
        max_length = int(CONFIG.get("max_length", 100))
        pad_mode = CONFIG.get("padding", "post")
        trunc_mode = CONFIG.get("truncating", "post")

        padded_seq = pad_sequences(
            sequences,
            maxlen=max_length,
            padding=pad_mode,
            truncating=trunc_mode
        )

        # Step 3: Neural network prediction
        raw_prediction = MODEL.predict(padded_seq, verbose=0)

        # Step 4: Handle output activation shape (sigmoid vs softmax)
        # Based on config.json, spam_rnn has 1 output unit with sigmoid activation
        classes = list(LABEL_ENCODER.classes_)  # ['ham', 'spam']
        threshold = float(CONFIG.get("threshold", 0.5))

        if raw_prediction.shape[-1] == 1:
            # Binary Sigmoid output: represents P(class index 1) -> P('spam')
            spam_prob = float(raw_prediction[0][0])
            ham_prob = float(1.0 - spam_prob)
            
            # Identify which index is spam
            spam_class_idx = classes.index("spam") if "spam" in classes else 1
            ham_class_idx = 1 - spam_class_idx

            if spam_class_idx == 1:
                is_spam = spam_prob >= threshold
                predicted_label = "spam" if is_spam else "ham"
                confidence = spam_prob if is_spam else ham_prob
            else:
                # In the rare event classes_ were inverted ['spam', 'ham']
                is_spam = (1.0 - spam_prob) >= threshold
                predicted_label = "spam" if is_spam else "ham"
                confidence = (1.0 - spam_prob) if is_spam else spam_prob
                # Align probabilities
                spam_prob, ham_prob = ham_prob, spam_prob
        else:
            # Categorical Softmax fallback
            probs = raw_prediction[0]
            pred_idx = int(np.argmax(probs))
            predicted_label = str(LABEL_ENCODER.inverse_transform([pred_idx])[0])
            is_spam = (predicted_label.lower() == "spam")
            
            spam_idx = classes.index("spam") if "spam" in classes else 1
            ham_idx = classes.index("ham") if "ham" in classes else 0
            spam_prob = float(probs[spam_idx])
            ham_prob = float(probs[ham_idx])
            confidence = float(probs[pred_idx])

        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return jsonify({
            "success": True,
            "predicted_label": predicted_label,
            "is_spam": bool(is_spam),
            "confidence": round(confidence, 4),
            "spam_probability": round(spam_prob, 4),
            "ham_probability": round(ham_prob, 4),
            "inference_time_ms": inference_time_ms,
            "threshold_used": threshold
        }), 200

    except Exception as e:
        # Prevent exposing internal stack traces in response
        return jsonify({
            "success": False,
            "error": "An internal error occurred during message classification."
        }), 500


if __name__ == "__main__":
    # Local development server fallback
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
