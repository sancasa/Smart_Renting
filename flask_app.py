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
    Type = str(request.form['Type'])
    Bedrooms = int(request.form['Bedrooms'])
    Bathrooms = int(request.form['Bathrooms'])
    Garages = int(request.form['Garages'])
    Neighborhood = str(request.form['Neighborhood'])
    Suites = int(request.form['Suites'])
    Furnished = str(request.form['Furnished'])
    features = np.array([[Type, Bedrooms, Bathrooms, Garages, Neighborhood, Suites, Furnished]])
    df = pd.DataFrame(features, columns = ['Type', 'Bedrooms', 'Bathrooms', 'Garages', 'Neighborhood', 'Suites', 'Furnished'])
    prediction = model.predict(df)
    output = np.expm1(prediction[0])
    # return render_template('index.html', prediction_text=f'Valor calculado do aluguel: R${round(float(output),0)}')
    return f"Calculated value: R${int(output)}"

if __name__ == "__main__":
    app.run()
