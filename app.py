from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
books = pd.read_csv("books.csv")

# Combine features (title + author + genre)
books["features"] = (
    books["title"] + " " + books["author"] + " " + books["genre"]
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(books["features"])
similarity_matrix = cosine_similarity(tfidf_matrix)

# Flask app
app = Flask(__name__)

# Simple HTML template
TEMPLATE = """
<!doctype html>
<title>Book Recommender</title>
<h1>📚 Book Recommender</h1>
<form method="post">
  <label>Enter a Book Title:</label><br>
  <input type="text" name="title" required>
  <button type="submit">Recommend</button>
</form>
{% if recommendations %}
  <h2>Recommendations:</h2>
  <ul>
    {% for r in recommendations %}
      <li>{{ r }}</li>
    {% endfor %}
  </ul>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def recommend():
    recommendations = []
    if request.method == "POST":
        title = request.form["title"]
        if title in books["title"].values:
            idx = books[books["title"] == title].index[0]
            scores = list(enumerate(similarity_matrix[idx]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
            recommendations = [books.iloc[i[0]]["title"] for i in scores]
        else:
            recommendations = ["❌ Book not found in database."]
    return render_template_string(TEMPLATE, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)

