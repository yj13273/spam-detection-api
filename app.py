import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
url = "https://raw.githubusercontent.com/yj13273/spam-detection-ml/main/spam_data.csv"
df = pd.read_csv(url, encoding='latin1')[["v1", "v2"]]
df.columns = ["Category", "Message"]

# Preprocess
df["Category"] = df["Category"].map({"ham": 1, "spam": 0})
X = df["Message"]
y = df["Category"].astype(int)

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict
def predict_mail(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "✅ HAM MAIL" if pred == 1 else "🚫 SPAM MAIL"

# Gradio UI
interface = gr.Interface(
    fn=predict_mail,
    inputs=gr.Textbox(lines=3, placeholder="Enter email content..."),
    outputs="text",
    title="Spam Detection Model",
    description="Enter an email message to check whether it's SPAM or HAM."
)

interface.launch()
