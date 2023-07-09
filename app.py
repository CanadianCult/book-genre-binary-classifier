from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from functions import *

app = Flask(__name__, static_url_path='/static')

# Load the trained model
model = tf.keras.models.load_model('book_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    description = data['description']
    predicted_genre = reviewBook(model, description)
    # Return the predicted genre as a JSON response
    response = {'genre': predicted_genre}
    return jsonify(response)

if __name__ == '__main__':
    app.run()
