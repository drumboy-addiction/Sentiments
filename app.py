from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__)



with open('model_classifier','r') as f:
    classifier = pickle.load(f)
with open('vectorizer_pickle','r') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():
    return
    

if __name__ == '__main__':
    app.run()
