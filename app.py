from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

filename = 'model.pkl'

# Load the pickled model
with open(filename, 'rb') as f:
    model = pickle.load(f)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.json
    # Make prediction
    print(data)
    prediction = model.predict(data['input'])
    # Return the prediction as JSON
    return jsonify({'prediction': prediction.tolist()})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True,host='192.168.0.103')
