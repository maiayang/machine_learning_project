from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

app = Flask(__name__)

wine = pd.read_csv('winequality_red.csv')

Predictors = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
              'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
              'density', 'pH', 'sulphates', 'alcohol']
TargetVariable = 'quality'

imputer = SimpleImputer(strategy='mean')
X = wine[Predictors].values
X_imputed = imputer.fit_transform(X)

scaler = MinMaxScaler(feature_range=(1, 10))  # Scale to 1-10 range
X_scaled = scaler.fit_transform(X_imputed)

RegModel = KNeighborsRegressor(n_neighbors=2)
y = wine[TargetVariable].values
y_scaled = ((y - y.min()) / (y.max() - y.min())) * 9 + 1  # Scale to 1-10 range
RegModel.fit(X_scaled, y_scaled)

@app.route('/')
def home():
    predictor_ranges = {}
    for predictor in Predictors:
        min_val = wine[predictor].min()
        max_val = wine[predictor].max()
        predictor_ranges[predictor] = (min_val, max_val)
    return render_template('index.html', predictor_ranges=predictor_ranges)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = []
        for predictor in Predictors:
            input_value = request.form.get(predictor)
            if input_value is not None and input_value.strip() != '':
                input_data.append(float(input_value))
            else:
                input_data.append(np.nan)
        input_data = np.array(input_data).reshape(1, -1)
        
        input_data_imputed = imputer.transform(input_data)
        
        input_data_scaled = scaler.transform(input_data_imputed)
        
        prediction_scaled = RegModel.predict(input_data_scaled)[0]
        
        prediction = ((prediction_scaled - 1) / 9) * (y.max() - y.min()) + y.min()
        
        return render_template('result.html', quality="{:.2f}".format(prediction))
    except Exception as e:
        error_message = str(e)
        return render_template('error.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
