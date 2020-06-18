from flask import Flask, request, render_template
import pickle
app = Flask(__name__)



with open('Model/model.pkl','rb') as f:
    model = pickle.load(f)
with open('Model/transformer.pkl','rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def index():
    word = 'good'
    vectorized = vectorizer.transform([word])
    prediction = model.predict(vectorized)
    print(prediction)
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():
    return
    

if __name__ == '__main__':
    app.run()
