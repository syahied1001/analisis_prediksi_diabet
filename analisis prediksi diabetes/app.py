from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    age = int(data.get('age', 0))
    gender = data.get('gender', '')
    smoker = data.get('smoker', '')
    blood_pressure = int(data.get('bloodPressure', 0))
    
    risk = "Rendah"
    if age > 45 and (smoker == "ya" or blood_pressure > 130):
        risk = "Tinggi"
    elif age > 30 and (smoker == "ya" or blood_pressure > 120):
        risk = "Sedang"
    
    return jsonify({'risk': risk})

if __name__ == '__main__':
    app.run(debug=True)
