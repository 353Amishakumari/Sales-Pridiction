from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Form se data lena
        experience = float(request.form["experience"])
        marketing = float(request.form["marketing_spend"])
        store_size = float(request.form["store_size"])

        # Model input
        features = np.array([[experience, marketing, store_size]])
        prediction = model.predict(features)[0]

        return render_template("index.html", prediction=round(prediction, 2))

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)