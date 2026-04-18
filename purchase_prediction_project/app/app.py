
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open("models/model.pkl","rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    pred = model.predict([data])[0]
    return jsonify({"prediction": int(pred)})

app.run(debug=True)
