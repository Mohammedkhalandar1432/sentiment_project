from flask import Flask, render_template, request
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Initialize app
app = Flask(__name__)

# Load dataset
df = pd.read_csv("cleaned_twitter.csv.csv")

# Clean data
df = df.dropna(subset=['category'])
X = df['clean_text'].fillna("")
y = df['category']

# Vectorization (improved for accuracy)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB(alpha=0.5)
model.fit(X, y)

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        tweet = request.form["tweet"]

        # Transform input
        tweet_vector = vectorizer.transform([tweet])

        # Predict
        result = model.predict(tweet_vector)[0]

        # Convert label to text
        if result == 1:
            prediction = "Positive 😊"
        elif result == 0:
            prediction = "Neutral 😐"
        else:
            prediction = "Negative 😡"

    return render_template("index.html", prediction=prediction)

# Run app
if __name__ == "__main__":
    app.run(debug=True)