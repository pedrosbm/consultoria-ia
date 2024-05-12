from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app, origins='*')

with open("./CreditAi.pkl", "rb") as arquivo:
    modelo = pickle.load(arquivo)


@app.route("/prever", methods=["POST"])
def prever():
    dados = request.json

    previsao = modelo.predict([dados["features"]])

    return jsonify({"previsao": previsao.tolist()})

if __name__ == "__main__":
    app.run(debug=True, port=8080)