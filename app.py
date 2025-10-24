from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        age = request.form['age']
        job = request.form['job']
        marital = request.form['marital']
        education = request.form['education']
        default = request.form['default']
        balance = request.form['balance']
        housing = request.form['housing']
        loan = request.form['loan']
        duration = request.form['duration']
        
        # Process the form data here (you can save to database, perform calculations, etc.)
        print(age,job,marital,education,default,balance,housing,loan,duration)
        model = joblib.load('logistic_regression_model_0.8613172.pkl')
        input = np.array([[age,job,marital,education,default,balance,housing,loan,duration]],dtype=float)
        y = model.predict(input)
        print(y)
        return render_template('result.html', p=y[0])
        
        #return render_template('thankyou.html', age=age, job=job, marital=marital, education=education,default=default, balance=balance, housing=housing, loan=loan, duration=duration)
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
