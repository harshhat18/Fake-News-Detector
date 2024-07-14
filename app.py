from flask import Flask, request, jsonify, render_template, url_for
import joblib
import re

app = Flask(__name__)

# Load models and vectorizer
logistic_model = joblib.load("logistic_model.pkl")
dtc_model = joblib.load("dtc_model.pkl")
rfc_model = joblib.load("rfc_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# Preprocessing functionS
def wordopt(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d", "", text)
    text = re.sub(r"\n", " ", text)
    return text


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]
    model_choice = data["model"]

    # Preprocess the text
    processed_text = [wordopt(text)]
    transformed_text = vectorizer.transform(processed_text)

    # Model prediction
    if model_choice == "logistic":
        model = logistic_model
    elif model_choice == "decision_tree":
        model = dtc_model
    else:
        model = rfc_model

    prediction = model.predict(transformed_text)[0]

    return jsonify({"model_prediction": int(prediction)})


if __name__ == "__main__":
    app.run(debug=True)
