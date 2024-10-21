from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
 
app = Flask(__name__)

# loading the pickle data to a Pandas DataFrame
with open('heart_disease_data.pkl', 'rb') as f:
    heart_data = pickle.load(f)

# route for home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/about')
def about():
    return render_template('about.html')


# route for prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # get form data
    
    age1 = request.form['age']
    sex1 = request.form['sex']
    cp1 = request.form['cp']
    trestbps1 = request.form['trestbps']
    chol1 = request.form['chol']
    fbs1 = request.form['fbs']
    restecg1 = request.form['restecg']
    thalach1 = request.form['thalach']
    exang1 = request.form['exang']
    oldpeak1 = request.form['oldpeak']
    slope1 = request.form['slope']
    ca1 = request.form['ca']
    thal1 = request.form['thal']


    # preprocess data
    data = np.array([[age1, sex1, cp1, trestbps1, chol1, fbs1, restecg1, thalach1, exang1, oldpeak1, slope1, ca1, thal1]])
    data = pd.DataFrame(data, columns=heart_data.columns[:-1])


    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(heart_data.iloc[:, :-1], heart_data.iloc[:, -1], test_size=0.2, random_state=42)

    # fit models
    lr_model = pickle.load(open('lr_model.pkl', 'rb'))
    rf_model = pickle.load(open('rf_model.pkl', 'rb'))
    dt_model = pickle.load(open('dt_model.pkl', 'rb'))
    nb_model = pickle.load(open('nb_model.pkl', 'rb'))
    mlp_model = pickle.load(open('mlp_model.pkl', 'rb'))
    svc_model = pickle.load(open('svc_model.pkl', 'rb'))

    # make predictions
    lr_prediction = lr_model.predict(data)
    rf_prediction = rf_model.predict(data)
    dt_prediction = dt_model.predict(data)
    nb_prediction = nb_model.predict(data) 
    mlp_prediction = mlp_model.predict(data)
    svc_prediction = svc_model.predict(data)

    # count number of predicted values that are equal to 1
    num_ones = sum([lr_prediction[0], rf_prediction[0], dt_prediction[0], nb_prediction[0], mlp_prediction[0], svc_prediction[0]])

    # create dictionary to store results
    results = {
        'Logistic Regression': lr_prediction[0],
        'Random Forest': rf_prediction[0],
        'Decision Tree': dt_prediction[0],
        'Naive Bayes': nb_prediction[0],
        'MLP': mlp_prediction[0],
        'SVC': svc_prediction[0]
    }

    # add a new key-value pair to indicate presence or absence of heart disease
    if num_ones > 4:
        results['Heart Disease'] = 'High'
    elif num_ones >= 2:
        results['Heart Disease'] = 'Medium'
    else:
        results['Heart Disease'] = 'Low'

 


    return render_template('predict.html', results=results,age1=age1,sex1=sex1,cp1=cp1,trestbps1=trestbps1,chol1=chol1,fbs1=fbs1,restecg1=restecg1,thalach1=thalach1,exang1=exang1,oldpeak1=oldpeak1,slope1=slope1,ca1=ca1,thal1=thal1)

def compare_chest_pain():
    cp1 = int(request.form['cp']) 
   
    return render_template('predict.html', cp1=cp1)

def compare_Blood_pressure():
    trestbps1 = int(request.form['trestbps']) 
   
    return render_template('predict.html', trestbps1=trestbps1)

def compare_Cholesterol():
    chol1 = int(request.form['chol']) 
   
    return render_template('predict.html', chol1=chol1)

def compare_Blood_Sugar():
    fbs1 = int(request.form['fbs']) 
   
    return render_template('predict.html', fbs1=fbs1)

def compare_ECG():
    restecg1 = int(request.form['restecg']) 
   
    return render_template('predict.html', restecg1=restecg1)

def compare_Heart_Rate():
    thalach1 = int(request.form['thalach']) 
   
    return render_template('predict.html', thalach1=thalach1)

def compare_Angina():
    exang1 = int(request.form['exang']) 
   
    return render_template('predict.html', exang1=exang1)

def compare_ST_Depression():
    oldpeak1 = int(request.form['oldpeak']) 
   
    return render_template('predict.html', oldpeak1=oldpeak1)

def compare_Slope():
    slope1  = int(request.form['slope']) 
   
    return render_template('predict.html', slope1=slope1)

def compare_Vessels():
    ca1 = int(request.form['ca']) 
   
    return render_template('predict.html', ca1=ca1)

def compare_thal():
    thal1 = int(request.form['thal']) 
   
    return render_template('predict.html', thal1=thal1)



if __name__ == '__main__':
    app.run(debug=True)
    
