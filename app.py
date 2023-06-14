from flask import Flask, render_template, request, redirect
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('LRM.pkl', 'rb'))
car = pd.read_csv('Clean Car.csv')

url = "https://wise.com/tr/currency-converter/inr-to-usd-rate"
r = requests.get(url)
soup = BeautifulSoup(r.content, "html.parser")
gelen_veri = soup.findAll('span',{"class":"text-success"})
dolar_cek=gelen_veri[0].text

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    companies.insert(0, 'Select Company')

    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type,
                           dolar_cek=dolar_cek)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    km = request.form.get('kilomtr')

    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                            data=np.array([car_model, company, year, km, fuel_type]).reshape(1, 5)))

    return str(np.ceil(prediction[0]).astype(float))


if __name__ == '__main__':
    app.run()
