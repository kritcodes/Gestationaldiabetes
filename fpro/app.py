import numpy as np 
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__, template_folder='templates') 
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('diab.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    print(prediction)
   #checkk
    output ="Yes" if prediction[0] else "No"
    
    return render_template('diab.html', prediction_text='Is the person diagnosed with diabetes? {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)