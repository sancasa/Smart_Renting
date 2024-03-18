from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder='templates', static_folder='static') #
model = pickle.load(open('ml_model/model.pkl', 'rb'))

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    Tipo = str(request.form['Tipo'])
    Dormitórios = int(request.form['Dormitórios'])
    Banheiros = int(request.form['Banheiros'])
    Garagens = int(request.form['Garagens'])
    Bairro = str(request.form['Bairro'])
    Suítes = int(request.form['Suítes'])
    Mobiliado = str(request.form['Mobiliado'])
    features = np.array([[Tipo, Dormitórios, Banheiros, Garagens, Bairro, Suítes, Mobiliado]])
    df = pd.DataFrame(features, columns = ['Tipo', 'Dormitórios', 'Banheiros', 'Garagens', 'Bairro', 'Suítes', 'Mobiliado'])
    prediction = model.predict(df)
    output = np.expm1(prediction[0])
    # return render_template('index.html', prediction_text=f'Valor calculado do aluguel: R${round(float(output),0)}')
    return f"Valor calculado do aluguel: R${int(output)}"

if __name__ == "__main__":
    app.run()