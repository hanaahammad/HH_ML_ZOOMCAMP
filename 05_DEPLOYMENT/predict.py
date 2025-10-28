import pickle

from flask import Flask
from flask import request
from flask import jsonify


#model_file = 'model_C=1.0.bin'
model_file = 'pipeline_v1.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('converted')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    convert = y_pred >= 0.5
    '''
    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }
    '''
    result = {
        'convert_probability': float(y_pred),
        'convert': bool(convert)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=9696)