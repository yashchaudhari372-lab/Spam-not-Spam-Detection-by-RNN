"""
=================================================================================
  CineSense AI  —  RNN-Powered Movie Review Sentiment Analysis Dashboard
=================================================================================
  Backend   : Flask
  Model     : Keras Sequential RNN loaded via pickle  (Embedding -> SimpleRNN -> Dense)
  Frontend  : Single-file HTML/CSS/JS dashboard (no templates folder needed)
              - 6 selectable colour themes
              - Animated gauge / doughnut / trend charts (Chart.js)
              - Glassmorphism cards, gradient backgrounds, micro-animations
              - Live history, stats, model architecture panel

  Run:
      pip install -r requirements.txt
      python app.py
  Then open:  http://127.0.0.1:5000

  NOTE: Place your RNN_model.pkl file in the same folder as this app.py
        (or set MODEL_PATH via the MODEL_PATH env var).
=================================================================================
"""

import os
import re
import pickle
import logging
from datetime import datetime

import numpy as np
from flask import Flask, request, jsonify, Response

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "RNN_model.pkl"),
)
VOCAB_SIZE = 5000        # matches Embedding(input_dim=5000, ...)
MAX_LEN = 50             # matches Embedding(input_length=50)
INDEX_FROM = 3           # standard Keras IMDB offset (0=PAD,1=START,2=UNK)
MAX_HISTORY = 25

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("cinesense")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the RNN model (pickle)
# ---------------------------------------------------------------------------
model = None
model_load_error = None
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    log.info("RNN model loaded successfully from %s", MODEL_PATH)
except Exception as exc:  # noqa: BLE001
    model_load_error = str(exc)
    log.error("Failed to load model: %s", exc)


def get_model_architecture():
    """Return a JSON-friendly description of the loaded model's layers."""
    layers_info = []
    total_params = 0
    if model is not None:
        try:
            for layer in model.layers:
                try:
                    params = layer.count_params()
                except Exception:  # noqa: BLE001
                    params = 0
                total_params += params
                try:
                    out_shape = str(layer.output.shape)
                except Exception:  # noqa: BLE001
                    out_shape = "N/A"
                layers_info.append(
                    {
                        "name": layer.name,
                        "type": layer.__class__.__name__,
                        "output_shape": out_shape,
                        "params": int(params),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not introspect model layers: %s", exc)
    return {
        "layers": layers_info,
        "total_params": int(total_params),
        "vocab_size": VOCAB_SIZE,
        "max_len": MAX_LEN,
        "loaded": model is not None,
        "error": model_load_error,
    }


# ---------------------------------------------------------------------------
# Tokenizer — lazily loads Keras' IMDB word index (with an offline fallback
# hashing tokenizer so the app still works with no internet access)
# ---------------------------------------------------------------------------
_word_index = None
_word_index_source = "unavailable"


def _load_word_index():
    global _word_index, _word_index_source
    if _word_index is not None:
        return _word_index
    try:
        from tensorflow.keras.datasets import imdb

        _word_index = imdb.get_word_index()
        _word_index_source = "keras-imdb"
        log.info("Loaded official Keras IMDB word index (%d words).", len(_word_index))
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Could not fetch Keras IMDB word index (%s). Using fallback hashing tokenizer.",
            exc,
        )
        _word_index = {}
        _word_index_source = "hash-fallback"
    return _word_index


_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def encode_review(text: str, vocab_size: int = VOCAB_SIZE, maxlen: int = MAX_LEN):
    """Convert raw review text into a padded integer sequence the model expects."""
    word_index = _load_word_index()
    words = _TOKEN_RE.findall(text.lower())

    sequence = []
    for w in words:
        if _word_index_source == "keras-imdb":
            idx = word_index.get(w)
            if idx is None:
                token = 2  # <UNK>
            else:
                token = idx + INDEX_FROM
                if token >= vocab_size:
                    token = 2
        else:
            # Deterministic hashing fallback keeps behaviour stable without internet
            token = (abs(hash(w)) % (vocab_size - INDEX_FROM)) + INDEX_FROM
        sequence.append(token)

    # pad/truncate (pre-padding, matches typical Keras 'pad_sequences' default)
    if len(sequence) < maxlen:
        sequence = [0] * (maxlen - len(sequence)) + sequence
    else:
        sequence = sequence[-maxlen:]

    return np.array([sequence], dtype=np.int32), len(words)


# ---------------------------------------------------------------------------
# In-memory prediction history (demo purposes)
# ---------------------------------------------------------------------------
history = []


def add_to_history(entry):
    history.insert(0, entry)
    del history[MAX_HISTORY:]


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    return jsonify(get_model_architecture())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if model is None:
        return jsonify({"error": f"Model not loaded: {model_load_error}"}), 500

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter a movie review to analyze."}), 400
    if len(text) > 5000:
        return jsonify({"error": "Review is too long (max 5000 characters)."}), 400

    try:
        encoded, word_count = encode_review(text)
        raw_pred = model.predict(encoded, verbose=0)
        probability = float(np.ravel(raw_pred)[0])
        probability = max(0.0, min(1.0, probability))

        sentiment = "Positive" if probability >= 0.5 else "Negative"
        confidence = probability if sentiment == "Positive" else (1 - probability)

        result = {
            "sentiment": sentiment,
            "probability": round(probability, 4),
            "confidence": round(confidence * 100, 2),
            "word_count": word_count,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "preview": (text[:80] + "…") if len(text) > 80 else text,
        }
        add_to_history(result)
        return jsonify(result)

    except Exception as exc:  # noqa: BLE001
        log.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.route("/api/history", methods=["GET"])
def api_history():
    return jsonify({"history": history})


@app.route("/api/history/clear", methods=["POST"])
def api_history_clear():
    history.clear()
    return jsonify({"status": "cleared"})


# ---------------------------------------------------------------------------
# Frontend — single-file dashboard
# ---------------------------------------------------------------------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>CineSense AI &mdash; RNN Sentiment Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  :root{
    --font-display:'Space Grotesk', 'Poppins', sans-serif;
    --font-body:'Inter', 'Poppins', sans-serif;
    --radius:18px;
    --transition: all .35s cubic-bezier(.4,0,.2,1);
  }

  /* ---------------- THEMES ---------------- */
  [data-theme="neon"]{
    --bg1:#0f0c29; --bg2:#302b63; --bg3:#24243e;
    --accent1:#00f5d4; --accent2:#ff206e; --accent3:#ffd23f;
    --card-bg:rgba(255,255,255,.06); --card-border:rgba(255,255,255,.14);
    --text-main:#f4f4fb; --text-dim:#b6b3d6;
    --pos:#00f5a0; --neg:#ff4d6d;
  }
  [data-theme="ocean"]{
    --bg1:#001220; --bg2:#013a63; --bg3:#01497c;
    --accent1:#48cae4; --accent2:#00b4d8; --accent3:#90e0ef;
    --card-bg:rgba(255,255,255,.06); --card-border:rgba(144,224,239,.18);
    --text-main:#eaf6fb; --text-dim:#a9c9d9;
    --pos:#38f9d7; --neg:#ff6b6b;
  }
  [data-theme="sunset"]{
    --bg1:#1a0933; --bg2:#4a1942; --bg3:#8a3a4b;
    --accent1:#ff9e5e; --accent2:#ff5e78; --accent3:#ffd166;
    --card-bg:rgba(255,255,255,.07); --card-border:rgba(255,209,102,.18);
    --text-main:#fff3e8; --text-dim:#e0b8b0;
    --pos:#7ee787; --neg:#ff6f6f;
  }
  [data-theme="royal"]{
    --bg1:#0d0221; --bg2:#2b0a4a; --bg3:#3d0e6b;
    --accent1:#c77dff; --accent2:#7b2ff7; --accent3:#f72585;
    --card-bg:rgba(255,255,255,.06); --card-border:rgba(199,125,255,.2);
    --text-main:#f2e9ff; --text-dim:#c5b3e0;
    --pos:#4ade80; --neg:#f472b6;
  }
  [data-theme="forest"]{
    --bg1:#04160f; --bg2:#0b3d2e; --bg3:#155e46;
    --accent1:#2dd881; --accent2:#a3e635; --accent3:#facc15;
    --card-bg:rgba(255,255,255,.06); --card-border:rgba(163,230,53,.18);
    --text-main:#e9fdf3; --text-dim:#a9d4bf;
    --pos:#4ade80; --neg:#fb7185;
  }
  [data-theme="light"]{
    --bg1:#eef2ff; --bg2:#e0e7ff; --bg3:#f5f3ff;
    --accent1:#6366f1; --accent2:#ec4899; --accent3:#f59e0b;
    --card-bg:rgba(255,255,255,.75); --card-border:rgba(99,102,241,.18);
    --text-main:#1e1b3a; --text-dim:#5c5a7a;
    --pos:#16a34a; --neg:#dc2626;
  }

  *{box-sizing:border-box; margin:0; padding:0;}
  html,body{height:100%;}
  body{
    font-family:var(--font-body);
    color:var(--text-main);
    background:linear-gradient(-45deg, var(--bg1), var(--bg2), var(--bg3), var(--bg1));
    background-size:400% 400%;
    animation:gradientShift 18s ease infinite;
    min-height:100vh;
    transition:var(--transition);
    overflow-x:hidden;
  }
  @keyframes gradientShift{
    0%{background-position:0% 50%;}
    50%{background-position:100% 50%;}
    100%{background-position:0% 50%;}
  }

  /* floating orbs */
  .orb{position:fixed; border-radius:50%; filter:blur(60px); opacity:.35; z-index:0; pointer-events:none;}
  .orb1{width:380px;height:380px; background:var(--accent1); top:-100px; left:-100px; animation:float1 12s ease-in-out infinite;}
  .orb2{width:320px;height:320px; background:var(--accent2); bottom:-80px; right:-80px; animation:float2 14s ease-in-out infinite;}
  .orb3{width:260px;height:260px; background:var(--accent3); top:40%; right:10%; animation:float1 16s ease-in-out infinite;}
  @keyframes float1{0%,100%{transform:translate(0,0);}50%{transform:translate(40px,60px);}}
  @keyframes float2{0%,100%{transform:translate(0,0);}50%{transform:translate(-50px,-40px);}}

  .wrap{position:relative; z-index:1; max-width:1320px; margin:0 auto; padding:28px 24px 60px;}

  /* ---------------- HEADER ---------------- */
  header{
    display:flex; align-items:center; justify-content:space-between;
    gap:16px; flex-wrap:wrap; margin-bottom:28px;
  }
  .brand{display:flex; align-items:center; gap:14px;}
  .brand-icon{
    width:52px; height:52px; border-radius:16px;
    background:linear-gradient(135deg, var(--accent1), var(--accent2));
    display:flex; align-items:center; justify-content:center;
    font-size:26px; box-shadow:0 8px 24px -6px var(--accent2);
    animation:pulseIcon 3s ease-in-out infinite;
  }
  @keyframes pulseIcon{0%,100%{transform:scale(1);}50%{transform:scale(1.07);}}
  .brand h1{
    font-family:var(--font-display); font-weight:700; font-size:26px;
    background:linear-gradient(90deg, var(--accent1), var(--accent2), var(--accent3));
    -webkit-background-clip:text; background-clip:text; color:transparent;
    background-size:200% auto; animation:shine 5s linear infinite;
  }
  @keyframes shine{to{background-position:200% center;}}
  .brand p{font-size:12.5px; color:var(--text-dim); letter-spacing:.5px; margin-top:2px;}

  .theme-picker{display:flex; align-items:center; gap:10px; background:var(--card-bg); border:1px solid var(--card-border); padding:8px 14px; border-radius:100px; backdrop-filter:blur(14px);}
  .theme-picker span{font-size:12px; color:var(--text-dim); font-weight:600; letter-spacing:.4px;}
  .swatch{
    width:26px; height:26px; border-radius:50%; cursor:pointer; border:2px solid rgba(255,255,255,.25);
    transition:var(--transition); position:relative;
  }
  .swatch:hover{transform:scale(1.18);}
  .swatch.active{border-color:#fff; box-shadow:0 0 0 3px rgba(255,255,255,.18);}
  .swatch[data-t="neon"]{background:linear-gradient(135deg,#00f5d4,#ff206e);}
  .swatch[data-t="ocean"]{background:linear-gradient(135deg,#48cae4,#01497c);}
  .swatch[data-t="sunset"]{background:linear-gradient(135deg,#ff9e5e,#ff5e78);}
  .swatch[data-t="royal"]{background:linear-gradient(135deg,#c77dff,#7b2ff7);}
  .swatch[data-t="forest"]{background:linear-gradient(135deg,#2dd881,#155e46);}
  .swatch[data-t="light"]{background:linear-gradient(135deg,#6366f1,#f59e0b);}

  /* ---------------- STAT CARDS ---------------- */
  .stats{display:grid; grid-template-columns:repeat(4,1fr); gap:18px; margin-bottom:26px;}
  .stat-card{
    background:var(--card-bg); border:1px solid var(--card-border); border-radius:var(--radius);
    padding:18px 20px; backdrop-filter:blur(16px); transition:var(--transition);
    position:relative; overflow:hidden;
  }
  .stat-card:hover{transform:translateY(-4px); box-shadow:0 14px 30px -12px rgba(0,0,0,.4);}
  .stat-card .label{font-size:12px; color:var(--text-dim); letter-spacing:.5px; font-weight:600; text-transform:uppercase;}
  .stat-card .value{font-family:var(--font-display); font-size:30px; font-weight:700; margin-top:6px;}
  .stat-card .bar{height:4px; border-radius:4px; margin-top:12px; background:rgba(255,255,255,.1); overflow:hidden;}
  .stat-card .bar i{display:block; height:100%; border-radius:4px; transition:width 1s cubic-bezier(.4,0,.2,1);}

  /* ---------------- MAIN GRID ---------------- */
  .grid{display:grid; grid-template-columns:1.1fr .9fr; gap:22px;}
  @media(max-width:980px){.grid{grid-template-columns:1fr;} .stats{grid-template-columns:repeat(2,1fr);}}

  .card{
    background:var(--card-bg); border:1px solid var(--card-border); border-radius:var(--radius);
    padding:24px; backdrop-filter:blur(16px); transition:var(--transition);
    box-shadow:0 8px 30px -14px rgba(0,0,0,.35);
  }
  .card h2{font-family:var(--font-display); font-size:17px; font-weight:600; margin-bottom:16px; display:flex; align-items:center; gap:8px;}
  .card h2 .dot{width:8px;height:8px;border-radius:50%;background:var(--accent1); box-shadow:0 0 10px var(--accent1);}

  textarea{
    width:100%; min-height:150px; resize:vertical; border-radius:14px; padding:16px;
    background:rgba(0,0,0,.18); border:1px solid var(--card-border); color:var(--text-main);
    font-family:var(--font-body); font-size:14.5px; line-height:1.6; outline:none; transition:var(--transition);
  }
  [data-theme="light"] textarea{background:rgba(255,255,255,.6);}
  textarea:focus{border-color:var(--accent1); box-shadow:0 0 0 3px color-mix(in srgb, var(--accent1) 25%, transparent);}
  textarea::placeholder{color:var(--text-dim);}

  .toolbar{display:flex; justify-content:space-between; align-items:center; margin-top:10px; font-size:12px; color:var(--text-dim);}

  .btn{
    margin-top:16px; width:100%; padding:15px; border:none; border-radius:14px; cursor:pointer;
    font-family:var(--font-display); font-weight:600; font-size:15px; letter-spacing:.3px;
    background:linear-gradient(120deg, var(--accent1), var(--accent2));
    color:#0b0b12; position:relative; overflow:hidden; transition:var(--transition);
    box-shadow:0 10px 26px -10px var(--accent2);
  }
  .btn:hover{transform:translateY(-2px); box-shadow:0 16px 32px -10px var(--accent2);}
  .btn:active{transform:translateY(0);}
  .btn:disabled{opacity:.6; cursor:not-allowed; transform:none;}
  .btn .spinner{width:16px;height:16px;border-radius:50%;border:2.5px solid rgba(0,0,0,.25); border-top-color:#0b0b12; display:none; vertical-align:middle; margin-right:8px; animation:spin .7s linear infinite;}
  .btn.loading .spinner{display:inline-block;}
  .btn.loading .btn-label{opacity:.85;}
  @keyframes spin{to{transform:rotate(360deg);}}

  .examples{display:flex; flex-wrap:wrap; gap:8px; margin-top:14px;}
  .chip{
    font-size:12px; padding:7px 13px; border-radius:100px; cursor:pointer;
    background:rgba(255,255,255,.07); border:1px solid var(--card-border); color:var(--text-dim);
    transition:var(--transition);
  }
  .chip:hover{background:var(--accent1); color:#0b0b12; border-color:transparent;}

  /* ---------------- RESULT PANEL ---------------- */
  .result-empty{display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding:40px 10px; color:var(--text-dim);}
  .result-empty .emoji{font-size:46px; margin-bottom:12px; opacity:.7;}

  .result-body{display:none;}
  .result-body.show{display:block; animation:fadeUp .5s ease;}
  @keyframes fadeUp{from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:translateY(0);}}

  .sentiment-badge{
    display:inline-flex; align-items:center; gap:8px; padding:9px 18px; border-radius:100px;
    font-family:var(--font-display); font-weight:700; font-size:15px; margin-bottom:18px;
  }
  .sentiment-badge.pos{background:color-mix(in srgb, var(--pos) 22%, transparent); color:var(--pos); box-shadow:0 0 0 1px color-mix(in srgb, var(--pos) 45%, transparent);}
  .sentiment-badge.neg{background:color-mix(in srgb, var(--neg) 22%, transparent); color:var(--neg); box-shadow:0 0 0 1px color-mix(in srgb, var(--neg) 45%, transparent);}
  .sentiment-badge .pulse{width:9px;height:9px;border-radius:50%; background:currentColor; animation:pulseDot 1.4s ease-in-out infinite;}
  @keyframes pulseDot{0%,100%{opacity:1; transform:scale(1);} 50%{opacity:.4; transform:scale(1.5);}}

  .gauge-wrap{display:flex; align-items:center; gap:22px; flex-wrap:wrap;}
  .gauge-canvas-holder{position:relative; width:170px; height:170px;}
  .gauge-center{position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;}
  .gauge-center .pct{font-family:var(--font-display); font-size:28px; font-weight:700;}
  .gauge-center .pct-label{font-size:10.5px; color:var(--text-dim); letter-spacing:.5px; text-transform:uppercase;}

  .metric-list{flex:1; min-width:180px; display:flex; flex-direction:column; gap:12px;}
  .metric-row{display:flex; justify-content:space-between; align-items:center; font-size:13px; color:var(--text-dim);}
  .metric-row b{color:var(--text-main); font-family:var(--font-display); font-size:14px;}

  .confidence-track{height:10px; border-radius:10px; background:rgba(255,255,255,.1); overflow:hidden; margin-top:6px;}
  .confidence-fill{height:100%; border-radius:10px; width:0%; transition:width 1.1s cubic-bezier(.4,0,.2,1); background:linear-gradient(90deg, var(--accent1), var(--accent2));}

  /* ---------------- CHARTS ROW ---------------- */
  .charts-row{display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:22px;}
  @media(max-width:600px){.charts-row{grid-template-columns:1fr;}}
  .chart-box{height:190px; position:relative;}

  /* ---------------- HISTORY ---------------- */
  .history-list{display:flex; flex-direction:column; gap:10px; max-height:280px; overflow-y:auto; padding-right:4px;}
  .history-list::-webkit-scrollbar{width:6px;}
  .history-list::-webkit-scrollbar-thumb{background:var(--accent1); border-radius:10px;}
  .hist-item{
    display:flex; align-items:center; gap:12px; padding:12px 14px; border-radius:12px;
    background:rgba(255,255,255,.05); border:1px solid var(--card-border); font-size:13px;
    animation:fadeUp .4s ease;
  }
  .hist-badge{width:8px; height:8px; border-radius:50%; flex-shrink:0;}
  .hist-badge.pos{background:var(--pos); box-shadow:0 0 8px var(--pos);}
  .hist-badge.neg{background:var(--neg); box-shadow:0 0 8px var(--neg);}
  .hist-text{flex:1; color:var(--text-dim); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
  .hist-pct{font-family:var(--font-display); font-weight:700; font-size:12.5px;}
  .hist-time{font-size:11px; color:var(--text-dim); opacity:.7;}
  .empty-hist{text-align:center; color:var(--text-dim); font-size:13px; padding:30px 0;}

  /* ---------------- MODEL INFO ---------------- */
  .arch-row{display:flex; align-items:center; gap:10px; margin-bottom:10px;}
  .arch-pill{
    flex:1; padding:12px 14px; border-radius:12px; background:rgba(255,255,255,.05);
    border:1px solid var(--card-border); display:flex; justify-content:space-between; align-items:center;
  }
  .arch-pill .n{font-family:var(--font-display); font-weight:600; font-size:13px;}
  .arch-pill .t{font-size:11px; color:var(--text-dim);}
  .arch-pill .p{font-size:11px; color:var(--accent1); font-weight:600;}
  .arrow{color:var(--text-dim); font-size:16px;}

  footer{text-align:center; margin-top:36px; font-size:12px; color:var(--text-dim);}
  footer b{color:var(--accent1);}

  .toast-wrap{position:fixed; bottom:24px; right:24px; z-index:50; display:flex; flex-direction:column; gap:10px;}
  .toast{
    background:var(--card-bg); border:1px solid var(--card-border); backdrop-filter:blur(16px);
    padding:13px 18px; border-radius:12px; font-size:13px; color:var(--text-main);
    box-shadow:0 10px 30px -10px rgba(0,0,0,.5); animation:toastIn .35s ease, toastOut .4s ease 2.6s forwards;
    border-left:3px solid var(--neg);
  }
  @keyframes toastIn{from{opacity:0; transform:translateX(30px);} to{opacity:1; transform:translateX(0);}}
  @keyframes toastOut{to{opacity:0; transform:translateX(30px);}}
</style>
</head>
<body data-theme="neon">
<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div class="orb orb3"></div>

<div class="wrap">

  <header>
    <div class="brand">
      <div class="brand-icon">🎬</div>
      <div>
        <h1>CineSense AI</h1>
        <p>RNN-Powered Movie Review Sentiment Intelligence</p>
      </div>
    </div>
    <div class="theme-picker">
      <span>THEME</span>
      <div class="swatch" data-t="neon" title="Midnight Neon"></div>
      <div class="swatch" data-t="ocean" title="Ocean Breeze"></div>
      <div class="swatch" data-t="sunset" title="Sunset Vibes"></div>
      <div class="swatch" data-t="royal" title="Royal Purple"></div>
      <div class="swatch" data-t="forest" title="Forest Emerald"></div>
      <div class="swatch" data-t="light" title="Classic Light"></div>
    </div>
  </header>

  <section class="stats">
    <div class="stat-card">
      <div class="label">Total Analyzed</div>
      <div class="value" id="statTotal">0</div>
      <div class="bar"><i style="width:0%; background:var(--accent1);" id="barTotal"></i></div>
    </div>
    <div class="stat-card">
      <div class="label">Positive Reviews</div>
      <div class="value" id="statPos" style="color:var(--pos);">0</div>
      <div class="bar"><i style="width:0%; background:var(--pos);" id="barPos"></i></div>
    </div>
    <div class="stat-card">
      <div class="label">Negative Reviews</div>
      <div class="value" id="statNeg" style="color:var(--neg);">0</div>
      <div class="bar"><i style="width:0%; background:var(--neg);" id="barNeg"></i></div>
    </div>
    <div class="stat-card">
      <div class="label">Avg. Confidence</div>
      <div class="value" id="statConf">0%</div>
      <div class="bar"><i style="width:0%; background:var(--accent3);" id="barConf"></i></div>
    </div>
  </section>

  <div class="grid">
    <!-- LEFT COLUMN -->
    <div style="display:flex; flex-direction:column; gap:22px;">

      <div class="card">
        <h2><span class="dot"></span> Analyze a Review</h2>
        <textarea id="reviewInput" maxlength="5000" placeholder="Paste or type a movie review here... e.g. &quot;This film was an absolute masterpiece with stunning visuals and a gripping story.&quot;"></textarea>
        <div class="toolbar">
          <span id="charCount">0 / 5000 characters</span>
          <span id="wordCount">0 words</span>
        </div>
        <button class="btn" id="analyzeBtn">
          <span class="spinner"></span><span class="btn-label">✨ Analyze Sentiment</span>
        </button>
        <div class="examples">
          <div class="chip" data-txt="This movie was absolutely fantastic, the acting was superb and the story kept me hooked till the very end.">😍 Rave review</div>
          <div class="chip" data-txt="Terrible film, boring plot, wasted two hours of my life and the acting was wooden.">😡 Harsh review</div>
          <div class="chip" data-txt="It was okay, some good moments but overall pretty forgettable and slow in the middle.">😐 Mixed review</div>
        </div>
      </div>

      <div class="card">
        <h2><span class="dot" style="background:var(--accent2); box-shadow:0 0 10px var(--accent2);"></span> Prediction Result</h2>
        <div class="result-empty" id="resultEmpty">
          <div class="emoji">🎯</div>
          <div>Enter a review above and click <b>Analyze Sentiment</b><br/>to see the AI's prediction.</div>
        </div>

        <div class="result-body" id="resultBody">
          <span class="sentiment-badge" id="sentimentBadge"><span class="pulse"></span><span id="sentimentText">Positive</span></span>

          <div class="gauge-wrap">
            <div class="gauge-canvas-holder">
              <canvas id="gaugeChart"></canvas>
              <div class="gauge-center">
                <div class="pct" id="gaugePct">0%</div>
                <div class="pct-label">Confidence</div>
              </div>
            </div>
            <div class="metric-list">
              <div class="metric-row">Raw model probability <b id="mProb">0.000</b></div>
              <div class="metric-row">Predicted class <b id="mClass">—</b></div>
              <div class="metric-row">Words processed <b id="mWords">0</b></div>
              <div class="metric-row">Analyzed at <b id="mTime">—</b></div>
              <div>
                <div class="metric-row" style="margin-bottom:0;"><span>Confidence level</span></div>
                <div class="confidence-track"><div class="confidence-fill" id="confFill"></div></div>
              </div>
            </div>
          </div>

          <div class="charts-row">
            <div class="chart-box"><canvas id="splitChart"></canvas></div>
            <div class="chart-box"><canvas id="trendChart"></canvas></div>
          </div>
        </div>
      </div>
    </div>

    <!-- RIGHT COLUMN -->
    <div style="display:flex; flex-direction:column; gap:22px;">

      <div class="card">
        <h2><span class="dot" style="background:var(--accent3); box-shadow:0 0 10px var(--accent3);"></span> Model Architecture</h2>
        <div id="archList">
          <div class="empty-hist">Loading model info…</div>
        </div>
      </div>

      <div class="card">
        <h2><span class="dot" style="background:var(--pos); box-shadow:0 0 10px var(--pos);"></span> Recent Predictions</h2>
        <div class="history-list" id="historyList">
          <div class="empty-hist">No predictions yet. Analyze a review to get started!</div>
        </div>
      </div>

    </div>
  </div>

  <footer>Built with <b>Flask</b> + <b>Keras RNN</b> · CineSense AI Dashboard &copy; <span id="year"></span></footer>
</div>

<div class="toast-wrap" id="toastWrap"></div>

<script>
document.getElementById('year').textContent = new Date().getFullYear();

/* ---------------- THEME SWITCHING ---------------- */
function setTheme(t){
  document.body.setAttribute('data-theme', t);
  document.querySelectorAll('.swatch').forEach(s=>s.classList.toggle('active', s.dataset.t===t));
  localStorage.setItem('cinesense-theme', t);
  updateChartTheme();
}
document.querySelectorAll('.swatch').forEach(sw=>{
  sw.addEventListener('click', ()=>setTheme(sw.dataset.t));
});

/* ---------------- TEXTAREA COUNTERS ---------------- */
const reviewInput = document.getElementById('reviewInput');
reviewInput.addEventListener('input', ()=>{
  const v = reviewInput.value;
  document.getElementById('charCount').textContent = v.length + ' / 5000 characters';
  const words = v.trim().length ? v.trim().split(/\s+/).length : 0;
  document.getElementById('wordCount').textContent = words + ' words';
});
document.querySelectorAll('.chip').forEach(chip=>{
  chip.addEventListener('click', ()=>{
    reviewInput.value = chip.dataset.txt;
    reviewInput.dispatchEvent(new Event('input'));
  });
});

/* ---------------- TOASTS ---------------- */
function showToast(msg){
  const wrap = document.getElementById('toastWrap');
  const el = document.createElement('div');
  el.className = 'toast';
  el.textContent = msg;
  wrap.appendChild(el);
  setTimeout(()=> el.remove(), 3100);
}

/* ---------------- CHARTS ---------------- */
let gaugeChart, splitChart, trendChart;
const trendData = { labels: [], values: [] };

function cssVar(name){ return getComputedStyle(document.body).getPropertyValue(name).trim(); }

function buildGauge(pct, isPos){
  const ctx = document.getElementById('gaugeChart');
  const color = isPos ? cssVar('--pos') : cssVar('--neg');
  if(gaugeChart) gaugeChart.destroy();
  gaugeChart = new Chart(ctx, {
    type: 'doughnut',
    data: { datasets: [{ data: [pct, 100-pct], backgroundColor: [color, 'rgba(255,255,255,.08)'], borderWidth: 0, cutout: '78%' }] },
    options: { responsive:true, maintainAspectRatio:false, animation:{ animateRotate:true, duration:1000 }, plugins:{ legend:{display:false}, tooltip:{enabled:false} } }
  });
}

function buildSplit(pos, neg){
  const ctx = document.getElementById('splitChart');
  if(splitChart) splitChart.destroy();
  splitChart = new Chart(ctx, {
    type: 'doughnut',
    data: { labels:['Positive','Negative'], datasets:[{ data:[pos, neg], backgroundColor:[cssVar('--pos'), cssVar('--neg')], borderWidth:0 }] },
    options: { responsive:true, maintainAspectRatio:false, plugins:{ legend:{ position:'bottom', labels:{ color: cssVar('--text-dim'), font:{ family:'Inter', size:11 }, usePointStyle:true, padding:12 } } } }
  });
}

function buildTrend(){
  const ctx = document.getElementById('trendChart');
  if(trendChart) trendChart.destroy();
  trendChart = new Chart(ctx, {
    type: 'line',
    data: { labels: trendData.labels, datasets:[{ label:'Confidence %', data: trendData.values, borderColor: cssVar('--accent1'), backgroundColor: 'transparent', tension:.35, pointRadius:3, pointBackgroundColor: cssVar('--accent2'), borderWidth:2.5 }] },
    options: { responsive:true, maintainAspectRatio:false,
      scales:{ y:{ min:0, max:100, ticks:{ color: cssVar('--text-dim'), font:{size:10} }, grid:{ color:'rgba(255,255,255,.06)'} },
               x:{ ticks:{ color: cssVar('--text-dim'), font:{size:10} }, grid:{ display:false } } },
      plugins:{ legend:{ display:false } } }
  });
}

function updateChartTheme(){
  if(splitChart){ splitChart.data.datasets[0].backgroundColor = [cssVar('--pos'), cssVar('--neg')]; splitChart.options.plugins.legend.labels.color = cssVar('--text-dim'); splitChart.update(); }
  if(trendChart){ trendChart.data.datasets[0].borderColor = cssVar('--accent1'); trendChart.data.datasets[0].pointBackgroundColor = cssVar('--accent2'); trendChart.options.scales.y.ticks.color = cssVar('--text-dim'); trendChart.options.scales.x.ticks.color = cssVar('--text-dim'); trendChart.update(); }
}

/* ---------------- STATS ---------------- */
let totalCount=0, posCount=0, negCount=0, confSum=0;
function refreshStats(){
  document.getElementById('statTotal').textContent = totalCount;
  document.getElementById('statPos').textContent = posCount;
  document.getElementById('statNeg').textContent = negCount;
  const avgConf = totalCount ? Math.round(confSum/totalCount) : 0;
  document.getElementById('statConf').textContent = avgConf + '%';
  document.getElementById('barTotal').style.width = '100%';
  document.getElementById('barPos').style.width = (totalCount? (posCount/totalCount*100):0) + '%';
  document.getElementById('barNeg').style.width = (totalCount? (negCount/totalCount*100):0) + '%';
  document.getElementById('barConf').style.width = avgConf + '%';
}

/* ---------------- MODEL INFO ---------------- */
async function loadModelInfo(){
  try{
    const res = await fetch('/api/model-info');
    const info = await res.json();
    const el = document.getElementById('archList');
    if(!info.loaded){ el.innerHTML = '<div class="empty-hist">⚠️ Model failed to load: '+ (info.error||'unknown error') +'</div>'; return; }
    if(!info.layers.length){ el.innerHTML = '<div class="empty-hist">No layer info available.</div>'; return; }
    el.innerHTML = info.layers.map((l,i)=>`
      <div class="arch-row">
        <div class="arch-pill">
          <div><div class="n">${l.name}</div><div class="t">${l.type} · ${l.output_shape}</div></div>
          <div class="p">${l.params.toLocaleString()} params</div>
        </div>
        ${i < info.layers.length-1 ? '<div class="arrow">→</div>' : ''}
      </div>`).join('') + `<div class="metric-row" style="margin-top:10px;"><span>Total parameters</span><b>${info.total_params.toLocaleString()}</b></div>
      <div class="metric-row"><span>Vocabulary size</span><b>${info.vocab_size.toLocaleString()}</b></div>
      <div class="metric-row"><span>Sequence length</span><b>${info.max_len}</b></div>`;
  }catch(e){ document.getElementById('archList').innerHTML = '<div class="empty-hist">Could not load model info.</div>'; }
}
loadModelInfo();

/* ---------------- HISTORY RENDER ---------------- */
function renderHistory(items){
  const el = document.getElementById('historyList');
  if(!items.length){ el.innerHTML = '<div class="empty-hist">No predictions yet. Analyze a review to get started!</div>'; return; }
  el.innerHTML = items.map(h=>`
    <div class="hist-item">
      <div class="hist-badge ${h.sentiment==='Positive'?'pos':'neg'}"></div>
      <div class="hist-text">${h.preview}</div>
      <div class="hist-pct" style="color:${h.sentiment==='Positive'?'var(--pos)':'var(--neg)'}">${h.confidence}%</div>
      <div class="hist-time">${h.timestamp}</div>
    </div>`).join('');
}

/* ---------------- ANALYZE ACTION ---------------- */
const analyzeBtn = document.getElementById('analyzeBtn');
analyzeBtn.addEventListener('click', async ()=>{
  const text = reviewInput.value.trim();
  if(!text){ showToast('⚠️ Please enter a review first.'); return; }

  analyzeBtn.classList.add('loading');
  analyzeBtn.disabled = true;

  try{
    const res = await fetch('/api/predict', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text})
    });
    const data = await res.json();
    if(!res.ok){ showToast('❌ ' + (data.error || 'Prediction failed')); return; }

    // reveal result
    document.getElementById('resultEmpty').style.display='none';
    const body = document.getElementById('resultBody');
    body.classList.remove('show'); void body.offsetWidth; body.classList.add('show');

    const isPos = data.sentiment === 'Positive';
    const badge = document.getElementById('sentimentBadge');
    badge.className = 'sentiment-badge ' + (isPos?'pos':'neg');
    document.getElementById('sentimentText').textContent = (isPos?'😊 Positive':'😞 Negative');

    document.getElementById('gaugePct').textContent = data.confidence + '%';
    document.getElementById('mProb').textContent = data.probability.toFixed(4);
    document.getElementById('mClass').textContent = data.sentiment;
    document.getElementById('mWords').textContent = data.word_count;
    document.getElementById('mTime').textContent = data.timestamp;
    document.getElementById('confFill').style.width = data.confidence + '%';
    document.getElementById('confFill').style.background = isPos ? 'linear-gradient(90deg, var(--pos), var(--accent1))' : 'linear-gradient(90deg, var(--neg), var(--accent2))';

    buildGauge(data.confidence, isPos);
    totalCount++; isPos ? posCount++ : negCount++; confSum += data.confidence;
    buildSplit(posCount, negCount);
    refreshStats();

    trendData.labels.push(data.timestamp);
    trendData.values.push(data.confidence);
    if(trendData.labels.length>10){ trendData.labels.shift(); trendData.values.shift(); }
    buildTrend();

    const hRes = await fetch('/api/history');
    const hData = await hRes.json();
    renderHistory(hData.history);

    showToast(isPos ? '✅ Positive sentiment detected!' : '⚠️ Negative sentiment detected.');
  }catch(e){
    showToast('❌ Network / server error.');
  }finally{
    analyzeBtn.classList.remove('loading');
    analyzeBtn.disabled = false;
  }
});

buildTrend();
setTheme(localStorage.getItem('cinesense-theme') || 'neon');
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return Response(INDEX_HTML, mimetype="text/html")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
