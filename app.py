from flask import Flask, request, render_template
import pickle
import time
from sklearn import svm
from sklearn.metrics import classification_report
app = Flask(__name__)

with open('Model/model.pkl','rb') as f:
    model = pickle.load(f)
with open('Model/transformer.pkl','rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():
    # Passed variable should be of the name **text**
    text = request.form['text']
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)
    result = "It is a " + str(prediction[0]) + " sentence"
    # value returned wull be of name **result**
    return render_template('result.html', result = result)
    

if __name__ == '__main__':
    app.run()
