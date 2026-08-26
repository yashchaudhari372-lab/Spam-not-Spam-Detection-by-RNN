import os
import pickle
import numpy as np
import tensorflow as tf

from flask import Flask, request, render_template_string
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# Flask Application
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "spam_rnn_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")

# ============================================================
# Model Configuration
# ============================================================
# The provided model expects:
# Input shape: (None, 100)
# Output shape: (None, 1)
# Output activation: sigmoid
#
# The provided tokenizer is used directly.
# No new tokenizer is created.
# No model is retrained.
# ============================================================

MAX_SEQUENCE_LENGTH = 100
THRESHOLD = 0.5

model = None
tokenizer = None
startup_error = None


# ============================================================
# Load Model and Tokenizer Once
# ============================================================

def load_resources():
    global model
    global tokenizer
    global startup_error

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "spam_rnn_model.keras was not found."
            )

        if not os.path.exists(TOKENIZER_PATH):
            raise FileNotFoundError(
                "tokenizer.pkl was not found."
            )

        # Load the already-trained model.
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        # Load the already-trained tokenizer.
        with open(TOKENIZER_PATH, "rb") as file:
            tokenizer = pickle.load(file)

        # Verify tokenizer compatibility.
        if not hasattr(tokenizer, "texts_to_sequences"):
            raise ValueError(
                "The uploaded tokenizer is not a valid Keras tokenizer."
            )

        # Verify model input shape.
        input_shape = model.input_shape

        if len(input_shape) != 2:
            raise ValueError(
                "Unexpected model input format."
            )

        if input_shape[1] != MAX_SEQUENCE_LENGTH:
            raise ValueError(
                "The model sequence length does not match the "
                "configured preprocessing length."
            )

        startup_error = None

    except Exception:
        # Never expose internal exceptions to users.
        model = None
        tokenizer = None

        startup_error = (
            "The machine-learning model could not be loaded. "
            "Please make sure spam_rnn_model.keras and tokenizer.pkl "
            "are present in the application folder."
        )


# Load once when the application starts.
load_resources()


# ============================================================
# HTML + CSS + JavaScript
# Everything is contained inside app.py.
# ============================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>Spam Message Detector</title>

    <style>

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            min-height: 100vh;

            font-family:
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                Roboto,
                Helvetica,
                Arial,
                sans-serif;

            background:
                radial-gradient(
                    circle at top left,
                    #e0e7ff 0%,
                    transparent 35%
                ),
                radial-gradient(
                    circle at bottom right,
                    #dbeafe 0%,
                    transparent 35%
                ),
                #f8fafc;

            color: #172033;

            display: flex;
            align-items: center;
            justify-content: center;

            padding: 25px;
        }

        .container {
            width: 100%;
            max-width: 720px;
        }

        .card {
            background: rgba(255, 255, 255, 0.97);

            border: 1px solid #e2e8f0;

            border-radius: 24px;

            padding: 42px;

            box-shadow:
                0 25px 60px rgba(15, 23, 42, 0.10),
                0 5px 20px rgba(15, 23, 42, 0.05);
        }

        .header {
            text-align: center;
            margin-bottom: 35px;
        }

        .icon {
            width: 68px;
            height: 68px;

            margin: 0 auto 18px;

            display: flex;
            align-items: center;
            justify-content: center;

            border-radius: 18px;

            background: linear-gradient(
                135deg,
                #4f46e5,
                #7c3aed
            );

            color: white;

            font-size: 30px;

            box-shadow:
                0 12px 25px rgba(79, 70, 229, 0.25);
        }

        h1 {
            font-size: 32px;
            font-weight: 750;
            color: #111827;
            margin-bottom: 8px;
        }

        .subtitle {
            color: #64748b;
            font-size: 15px;
        }

        .form-group {
            margin-bottom: 22px;
        }

        label {
            display: block;

            font-size: 15px;
            font-weight: 650;

            color: #334155;

            margin-bottom: 10px;
        }

        textarea {
            width: 100%;

            min-height: 180px;

            resize: vertical;

            border: 2px solid #e2e8f0;

            border-radius: 16px;

            padding: 17px;

            font-family: inherit;

            font-size: 15px;

            line-height: 1.6;

            color: #1e293b;

            background: white;

            outline: none;

            transition:
                border-color 0.2s ease,
                box-shadow 0.2s ease;
        }

        textarea::placeholder {
            color: #94a3b8;
        }

        textarea:focus {
            border-color: #6366f1;

            box-shadow:
                0 0 0 4px rgba(99, 102, 241, 0.10);
        }

        .button-row {
            display: flex;

            gap: 12px;

            margin-top: 18px;
        }

        button {
            border: none;

            border-radius: 13px;

            padding: 14px 22px;

            font-family: inherit;

            font-size: 15px;

            font-weight: 650;

            cursor: pointer;

            transition:
                transform 0.15s ease,
                box-shadow 0.15s ease,
                opacity 0.15s ease;
        }

        .check-btn {
            flex: 1;

            color: white;

            background: linear-gradient(
                135deg,
                #4f46e5,
                #7c3aed
            );

            box-shadow:
                0 10px 20px rgba(79, 70, 229, 0.20);
        }

        .clear-btn {
            background: #f1f5f9;
            color: #475569;
        }

        button:hover {
            transform: translateY(-1px);
        }

        button:active {
            transform: translateY(0);
        }

        button:disabled {
            cursor: not-allowed;
            opacity: 0.65;
            transform: none;
        }

        .result {
            margin-top: 30px;

            padding: 25px;

            border-radius: 18px;

            text-align: center;

            border: 1px solid #e2e8f0;

            background: #f8fafc;
        }

        .result-title {
            font-size: 13px;

            font-weight: 700;

            letter-spacing: 0.08em;

            text-transform: uppercase;

            color: #64748b;

            margin-bottom: 10px;
        }

        .prediction {
            font-size: 30px;

            font-weight: 800;

            margin-bottom: 8px;
        }

        .prediction.spam {
            color: #dc2626;
        }

        .prediction.ham {
            color: #16a34a;
        }

        .confidence {
            color: #475569;
            font-size: 15px;
        }

        .confidence strong {
            color: #1e293b;
        }

        .error {
            margin-top: 22px;

            padding: 15px 17px;

            border-radius: 12px;

            background: #fef2f2;

            border: 1px solid #fecaca;

            color: #b91c1c;

            font-size: 14px;

            line-height: 1.5;
        }

        .info {
            margin-top: 22px;

            padding: 14px 16px;

            border-radius: 12px;

            background: #eff6ff;

            border: 1px solid #bfdbfe;

            color: #1d4ed8;

            font-size: 13px;

            line-height: 1.5;
        }

        .footer {
            text-align: center;

            margin-top: 24px;

            color: #94a3b8;

            font-size: 12px;
        }

        @media (max-width: 600px) {

            body {
                padding: 14px;
                align-items: flex-start;
            }

            .card {
                padding: 27px 20px;

                border-radius: 20px;

                margin-top: 15px;
            }

            h1 {
                font-size: 27px;
            }

            .icon {
                width: 58px;
                height: 58px;

                font-size: 25px;
            }

            textarea {
                min-height: 160px;
            }

            .button-row {
                flex-direction: column;
            }

            button {
                width: 100%;
            }

            .prediction {
                font-size: 26px;
            }
        }

    </style>

</head>


<body>

<div class="container">

    <div class="card">

        <div class="header">

            <div class="icon">
                🛡️
            </div>

            <h1>
                Spam Message Detector
            </h1>

            <p class="subtitle">
                AI-powered SMS Spam Detection
            </p>

        </div>


        <form
            method="POST"
            action="/predict"
            id="predictionForm"
        >

            <div class="form-group">

                <label for="message">
                    Enter your message
                </label>

                <textarea
                    id="message"
                    name="message"
                    maxlength="5000"
                    placeholder="Type or paste an SMS message here..."
                    required
                >{{ message }}</textarea>

            </div>


            <div class="button-row">

                <button
                    type="submit"
                    class="check-btn"
                    id="checkButton"
                >
                    Check Message
                </button>

                <button
                    type="button"
                    class="clear-btn"
                    onclick="clearMessage()"
                >
                    Clear
                </button>

            </div>

        </form>


        {% if result %}

        <div class="result">

            <div class="result-title">
                Prediction Result
            </div>

            <div
                class="prediction {% if result == 'SPAM' %}spam{% else %}ham{% endif %}"
            >
                {{ result }}
            </div>

            <div class="confidence">
                Confidence:
                <strong>{{ confidence }}%</strong>
            </div>

        </div>

        {% endif %}


        {% if error %}

        <div class="error">
            {{ error }}
        </div>

        {% endif %}


        {% if startup_error %}

        <div class="error">
            {{ startup_error }}
        </div>

        {% endif %}


        <div class="info">
            Your message is analyzed using the trained RNN
            machine-learning model.
        </div>

    </div>


    <div class="footer">
        Spam Message Detector • TensorFlow + Flask
    </div>

</div>


<script>

    const form =
        document.getElementById("predictionForm");

    const button =
        document.getElementById("checkButton");


    form.addEventListener("submit", function () {

        button.disabled = true;

        button.textContent = "Checking...";

    });


    function clearMessage() {

        const textarea =
            document.getElementById("message");

        textarea.value = "";

        textarea.focus();

        const result =
            document.querySelector(".result");

        if (result) {
            result.remove();
        }

    }

</script>


</body>

</html>
"""


# ============================================================
# Home Page
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return render_template_string(
        HTML_TEMPLATE,

        message="",

        result=None,

        confidence=None,

        error=None,

        startup_error=startup_error
    )


# ============================================================
# Prediction
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    message = request.form.get("message", "")

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not isinstance(message, str):

        return render_template_string(
            HTML_TEMPLATE,

            message="",

            result=None,

            confidence=None,

            error="Invalid message input.",

            startup_error=startup_error
        )


    message = message.strip()


    if not message:

        return render_template_string(
            HTML_TEMPLATE,

            message="",

            result=None,

            confidence=None,

            error="Please enter a message before checking it.",

            startup_error=startup_error
        )


    if len(message) > 5000:

        return render_template_string(
            HTML_TEMPLATE,

            message=message[:5000],

            result=None,

            confidence=None,

            error="Please enter a message shorter than 5000 characters.",

            startup_error=startup_error
        )


    # --------------------------------------------------------
    # Check model availability
    # --------------------------------------------------------

    if model is None or tokenizer is None:

        return render_template_string(
            HTML_TEMPLATE,

            message=message,

            result=None,

            confidence=None,

            error=(
                "The spam detection model is currently unavailable. "
                "Please check the model and tokenizer files."
            ),

            startup_error=startup_error
        )


    try:

        # ----------------------------------------------------
        # Convert text into integer tokens.
        #
        # IMPORTANT:
        # The existing tokenizer.pkl is used directly.
        # No new tokenizer is created.
        # ----------------------------------------------------

        sequences = tokenizer.texts_to_sequences(
            [message]
        )


        # ----------------------------------------------------
        # Pad/truncate to the sequence length expected
        # by the trained model.
        #
        # Model input:
        # (None, 100)
        #
        # ----------------------------------------------------

        padded_sequence = pad_sequences(

            sequences,

            maxlen=MAX_SEQUENCE_LENGTH,

            padding="pre",

            truncating="pre",

            dtype="int32"
        )


        # ----------------------------------------------------
        # Run the trained RNN model.
        # ----------------------------------------------------

        prediction = model.predict(
            padded_sequence,
            verbose=0
        )


        # Convert model output into a single probability.
        probability = float(
            np.asarray(prediction).reshape(-1)[0]
        )


        if not np.isfinite(probability):

            raise ValueError(
                "Invalid model prediction."
            )


        probability = float(
            np.clip(probability, 0.0, 1.0)
        )


        # ----------------------------------------------------
        # Classification
        #
        # The model has a sigmoid output.
        #
        # >= 0.5 -> SPAM
        # < 0.5  -> NOT SPAM
        # ----------------------------------------------------

        if probability >= THRESHOLD:

            result = "SPAM"

            confidence = probability

        else:

            result = "NOT SPAM"

            confidence = 1.0 - probability


        confidence_percent = round(
            confidence * 100,
            2
        )


        return render_template_string(

            HTML_TEMPLATE,

            message=message,

            result=result,

            confidence=confidence_percent,

            error=None,

            startup_error=None
        )


    except Exception:

        # Never expose the internal exception.
        return render_template_string(

            HTML_TEMPLATE,

            message=message,

            result=None,

            confidence=None,

            error=(
                "Sorry, the message could not be analyzed. "
                "Please try again."
            ),

            startup_error=None
        )


# ============================================================
# Local Development
# ============================================================
#
# For local testing:
#
#     python app.py
#
# For Render production deployment:
#
#     gunicorn app:app
#
# Render will use the Gunicorn command instead of this block.
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
