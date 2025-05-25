from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import logging
from datetime import datetime
import csv

app = Flask(__name__)

# Konfigurasi logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Load model & scaler
model = load_model('model_gru.h5', compile=False)
scaler_X = joblib.load('scaler_X.save')
scaler_y = joblib.load('scaler_y.save')
last_59_data = np.load('last_59_data.npy')  # shape (59, 4)

# Validasi input
def validate_input(value, name):
    try:
        val = float(value)
        if val <= 0:
            raise ValueError(f"{name} harus lebih dari 0")
        return val
    except ValueError:
        raise ValueError(f"{name} harus berupa angka valid")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        open_price = validate_input(request.form['open'], 'Open')
        high_price = validate_input(request.form['high'], 'High')
        low_price  = validate_input(request.form['low'], 'Low')

        user_input = np.array([[open_price, high_price, low_price]])
        user_scaled = scaler_X.transform(user_input)
        history_features = last_59_data[:, :-1]
        input_sequence = np.vstack([history_features, user_scaled])
        X_sequence = input_sequence.reshape(1, 60, 3)

        y_pred_scaled = model.predict(X_sequence)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        pred_price = y_pred[0][0]

        # Log aktivitas
        logging.info(f"Input user - Open: {open_price}, High: {high_price}, Low: {low_price}")
        logging.info(f"Hasil prediksi: {pred_price}")

        # Simpan ke riwayat CSV
        with open('history.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), open_price, high_price, low_price, pred_price])

        return render_template('index.html', prediction=f'Harga Emas Prediksi: ${pred_price:.2f}')

    except Exception as e:
        logging.error(f"Error saat prediksi: {e}")
        return render_template('index.html', prediction=f'Error: {e}')

@app.route('/history')
def history():
    try:
        with open('history.csv', newline='') as file:
            reader = csv.reader(file)
            history_data = list(reader)
    except FileNotFoundError:
        history_data = []

    return render_template('history.html', history=history_data)

@app.route('/grafik')
def grafik():
    try:
        timestamps = []
        predictions = []

        with open('history.csv', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 5:
                    timestamps.append(row[0])
                    predictions.append(float(row[4]))

        return render_template('grafik.html', timestamps=timestamps, predictions=predictions)

    except Exception as e:
        logging.error(f"Error saat baca grafik: {e}")
        return f"Error: {e}"


if __name__ == '__main__':
    app.run(debug=True)
