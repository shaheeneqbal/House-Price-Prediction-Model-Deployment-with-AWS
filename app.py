from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        
        floors = int(request.form.get('floors'))
        bedrooms = int(request.form.get('bedrooms'))
        bathrooms = int(request.form.get('bathrooms'))
        sqft_living = int(request.form.get('sqft_living'))
        sqft_lot = int(request.form.get('sqft_lot'))
        sqft_above = int(request.form.get('sqft_above'))
        sqft_basement= int(request.form.get('sqft_basement'))
        view = int(request.form.get('view'))
        waterfront = int(request.form.get('waterfront'))
        condition = int(request.form.get('condition'))
        city = int(request.form.get('city'))
        statezip = int(request.form.get('statezip'))
        
        features = np.array([ floors, bedrooms, bathrooms, sqft_living, sqft_lot, sqft_above, sqft_basement, view, waterfront, condition, city, statezip]).reshape(1, -1)
        prediction = model.predict(features)
        pred = round(prediction[0],2)
        
        
        return render_template('result.html',result = str(pred))
    


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)

