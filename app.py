import numpy as np 
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)
model2 = pickle.load(open('Trained.pkl', 'rb'))

@app.route('/')
def home():
	return render_template('index.html')
@app.route('/Predict', methods=['Post'])
def predict():
	int_inputs = [[float(x) for x in request.form.values()]]
	array = np.array(int_inputs)
	prediction = model2.predict(array)
	output = round(prediction[0], 2)
	return render_template('index.html', prediction_num = 'Salary should be: {}'.format(output))

if __name__ == '__main__':
	app.run(debug=True)
