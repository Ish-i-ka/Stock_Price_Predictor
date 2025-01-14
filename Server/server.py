from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Initialize Flask app
app = Flask(_name_)

# Load and preprocess data
df = pd.read_csv('./Server/Dataset/stock prices.csv')
df = df.dropna()

# Define feature columns and target column
feature_columns = ['Open', 'High', 'Low', 'Volume']
target_column = 'Close'

X = df[feature_columns]
y = df[target_column]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Flask route for stock price prediction
@app.route('/predict', methods=['GET'])
def predict_stock():
    try:
        # Parse input parameters from request args
        open_price = float(request.args.get('open'))
        high_price = float(request.args.get('high'))
        low_price = float(request.args.get('low'))
        volume = float(request.args.get('volume'))

        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'Open': [open_price],
            'High': [high_price],
            'Low': [low_price],
            'Volume': [volume]
        })

        # Make prediction
        predicted_close = model.predict(input_data)[0]

        # Return response
        return jsonify({
            'predicted_close_price': predicted_close
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

# Run the Flask app
if _name_ == '_main_':
    app.run(debug=True)