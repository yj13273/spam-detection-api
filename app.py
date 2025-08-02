from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Optional: allows frontend from other domains to access this

# Load model and vectorizer
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    raise Exception(f"Error loading model or vectorizer: {e}")

# Token for simple authentication
AUTH_TOKEN = "your-secret-token"  # 🔐 Change this to something strong!

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Spam Detection API is running ✅"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    # Token-based auth
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer ") or auth_header.split(" ")[1] != AUTH_TOKEN:
        return jsonify({"error": "Unauthorized access"}), 401

    # Parse incoming JSON
    try:
        data = request.get_json(force=True)
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No input text provided"}), 400
    except:
        return jsonify({"error": "Invalid JSON format"}), 400

    # Predict
    try:
        features = vectorizer.transform([text])
        prediction = model.predict(features)[0]
        result = "HAM" if prediction == 1 else "SPAM"
        return jsonify({"prediction": result}), 200
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
