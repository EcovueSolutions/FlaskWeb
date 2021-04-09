from flask import Flask, request, render_template
import json
import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
model = pickle.load(open('Insurance_Model.pkl', 'rb'))

@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/getcharges',methods=['POST'])
def getcharges():
    print('Getting predictions')
    record = request.json
    age = record['age']
    print(age)
    sex = record['sex']
    bmi = record['bmi']
    children = record['children']
    smoker = record['smoker']
    region = record['region']
    lr = pickle.load(open('Insurance_Model.pkl','rb'))
    score = lr.predict([[age,sex,bmi,children,smoker,region]]).tolist()
    return str(score[0])


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Prediction should be $ {}'.format(output))


if __name__ == '__main__':
    app.run()
