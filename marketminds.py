from flask import Flask, render_template, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime, timedelta

app = Flask(__name__)

# Función para cargar datos y entrenar el modelo
def train_model():
    df = yf.download('AAPL', period='7d', interval='1m')

    # Manejar valores nulos
    df = df.ffill()

    # Escalar los datos
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Crear conjuntos de datos para entrenamiento
    X, y = [], []
    for i in range(len(data_scaled) - 60):
        X.append(data_scaled[i:i+60])
        y.append(data_scaled[i+60])

    X, y = np.array(X), np.array(y)

    # Definir el modelo de red neuronal
    model = Sequential([
        Dense(64, activation='relu', input_shape=(60, 1)),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Entrenar el modelo
    model.fit(X, y, epochs=20, verbose=1, batch_size=8)

    return model, scaler

# Entrenar el modelo al iniciar el servidor
model, scaler = train_model()

# Función para predecir los próximos minutos
def predict_next_minutes():
    df = yf.download('AAPL', period='1d', interval='1m')

    print(f"Downloaded data: {df}")

    if df.empty:
        print("No se pudieron obtener datos de Yahoo Finance")
        return jsonify({'error': 'No se pudieron obtener datos de Yahoo Finance'})

    # Verificar que hay suficientes datos
    if len(df) < 60:
        print("No hay suficientes datos históricos para la predicción")
        return jsonify({'error': 'No hay suficientes datos históricos para la predicción'})

    # Escalar datos
    last_data = scaler.transform(df['Close'].values.reshape(-1, 1))

    # Verificar si tiene la forma correcta
    if last_data.shape[0] < 60:
        print("Datos insuficientes para predicción")
        return jsonify({'error': 'Datos insuficientes para predicción'})

    last_data = last_data[-60:].reshape(1, 60, 1)  # Use the last 60 data points

    # Realizar la predicción
    predictions = model.predict(last_data)

    if predictions is None or len(predictions) == 0:
        return jsonify({'error': 'No se pudo generar la predicción'})

    # Convertir la predicción a lista
    predictions = predictions.reshape(-1, 1)  # Reshape to 2D array
    predicted_values = scaler.inverse_transform(predictions).tolist()

    # Obtener las etiquetas de tiempo para los próximos 60 minutos
    current_time = datetime.now()
    labels = [(current_time + timedelta(minutes=i+1)).strftime('%H:%M %p') for i in range(60)]

    # Obtener información adicional del stock
    stock_info = yf.Ticker('AAPL').info

    # print(f"Predicted Values: {predicted_values}")
    # print(f"Labels: {labels}")
    # print(f"Stock Info: {stock_info}")

    return jsonify({'predictions': predicted_values, 'labels': labels, 'stock_info': stock_info})

# Ruta principal
@app.route('/')
def index():
    result = predict_next_minutes().get_json()
    if 'error' in result:
        labels = []
        data = []
        stock_info = {}
    else:
        labels = result['labels']
        data = result['predictions']
        stock_info = result['stock_info']
    # print(f"Labels: {labels}")
    # print(f"Data: {data}")
    return render_template('index.html', labels=labels, data=data, stock_info=stock_info)

# Ruta de predicción
@app.route('/predict')
def predict():
    data = predict_next_minutes()
    return jsonify({'predictions': data})

# Nueva ruta para obtener una nueva predicción
@app.route('/get_new_prediction')
def get_new_prediction():
    result = predict_next_minutes().get_json()
    if 'error' in result:
        return jsonify({'error': result['error']})
    labels = result['labels']
    data = result['predictions']
    stock_info = result['stock_info']
    return jsonify({'labels': labels, 'data': data, 'stock_info': stock_info})

if __name__ == '__main__':
    app.run(debug=True)
