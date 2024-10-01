from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('electricity_price_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        consumption = float(request.form['consumption'])
        prediction = model.predict(np.array([[consumption]]))
        return render_template('index.html', prediction_text=f'Estimated Electricity Price: ${prediction[0]:.2f}')
    
if __name__ == '__main__':
    app.run(debug=True)
