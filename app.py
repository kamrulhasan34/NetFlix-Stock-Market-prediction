from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import streamlit as st
from flask_cors import CORS
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib  # For saving/loading the model


app = Flask(__name__)
CORS(app)  # Add this after initializing the Flask app

# Load data
file_path = r'C:\\Users\\kamrul\\Desktop\\Thesis\\RandomForest/Netflix Data - STOCK.xlsx'
data = pd.read_excel(file_path)

# Data Cleaning
data_cleaned = data.drop(columns=['Unnamed: 10', 'Unnamed: 11'], errors='ignore').dropna()

# Features and target
features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'

X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model (RandomForest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model for later use
joblib.dump(model, 'stock_model.pkl')

# Load the saved model for predictions
model = joblib.load('stock_model.pkl')
joblib.dump(model, 'stock_model.pkl')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    day = int(request.form['day'])
    month = int(request.form['month'])
    year = int(request.form['year'])

    # Prepare input data (For now, we're using random data as a placeholder)
    # You would typically get the actual stock data for the given date.
    volume = np.random.randint(100000, 1000000)

    # Using mean values of Open/High/Low for the prediction
    input_data = pd.DataFrame({
        'Open': [X['Open'].mean()],
        'High': [X['High'].mean()],
        'Low': [X['Low'].mean()],
        'Volume': [volume]
    })

    # Predict the closing price
    predicted_close = model.predict(input_data)[0]

    # Return the prediction
    return jsonify({
        'open': input_data['Open'][0],
        'high': input_data['High'][0],
        'low': input_data['Low'][0],
        'close': predicted_close
    })

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)



    
