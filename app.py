from flask import Flask, request, url_for, redirect, render_template

import pickle
import numpy as np



model = pickle.load(open('models/framingham.pickle', 'rb'))
model1 = pickle.load(open('models/combine_heart.pickle', 'rb'))
model2 = pickle.load(open('models/diabetes_prediction_rf.pickle', 'rb'))


app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/heart")
def heart():
    return render_template('heart.html')

@app.route("/heartframg", methods=['GET'])
def heartf():
    return render_template("heartfram.html")

@app.route("/heartcombine", methods=['GET'])
def heartc():
    return render_template("heartcombined.html")

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/healthtips')
def healthtips():
    return render_template('healthtips.html')



@app.route("/heartfram", methods=['GET','POST'])
def heartfram():
    if request.method == 'POST':

        gender = int(request.form['gender'])
        if gender == 'Male':
            gender = 1
        else:
            gender = 0
        print(gender)

        age = int(request.form['age'])
        print()
        smoker = request.form['smoker']
        if smoker == 'Yes':
            smoker = 1
        else:
            smoker = 0

        print(smoker)
        cigs = int(request.form['cigs'])
        bp_meds = int(request.form['bp_meds'])
        if bp_meds == 'Yes':
            bp_meds = 1
        else:
            bp_meds = 0
        stroke = int(request.form['stroke'])
        if stroke == 'Yes':
            stroke = 1
        else:
            stroke = 0
        hyp = int(request.form['hyp'])
        if hyp == 'Yes':
            hyp = 1
        else:
            hyp = 0
        dia = 1
        chol = int(request.form['chol'])
        sysBp = int(request.form['sysBp'])
        diaBp = int(request.form['diaBp'])
        height = float(request.form['height'])
        weight = int(request.form['weight'])
        bmi = weight / (height * height)
        rate = int(request.form['rate'])
        glu = float(request.form['glu'])

        prediction = model.predict(np.array([gender, age, smoker, cigs, bp_meds, stroke, hyp, dia, chol, sysBp, diaBp, bmi, rate, glu]).reshape((1, -1)))
        output = round(prediction[0])
        if output == 0:
            return render_template('result.html',prediction="Congratulations, you are not affected with heart disease. Have a good diet.!!")
        else:
            return render_template('result.html',prediction="Sorry ! But you have been affected with heart disease. Please consult doctore to avoid losses.".format(output))

    else:
        return render_template('heartfram.html')


@app.route('/heartcombined', methods=['GET','POST'])
def heartcombined():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        if sex == 'Male':
            sex = 1
        else:
            sex = 0
        cpt = int(request.form['cpt'])
        if cpt == "Typical Angina":
            cpt = 0
        elif cpt == "Atypical Angina":
            cpt = 1
        elif cpt == "Non-Anginal Pain":
            cpt = 2
        else:
            cpt = 3
        bp = int(request.form['bp'])
        chol = int(request.form['chol'])
        fbp = int(request.form['fbp'])
        if fbp == "Fasting Blood Sugar < 120 mg/dl":
            fbp = 0
        else:
            fbp = 1
        ecg = int(request.form['ecg'])
        if ecg == "Recting Ecg":
            ecg = 0
        elif ecg == "ST-T wave abnormality":
            ecg = 1
        else:
            ecg = 2
        mhr = int(request.form['mhr'])
        mhr = 220 - age
        exe_angina = int(request.form['exe_angina'])
        if cpt == 4:
            exe_angina = 1
        else:
            exe_angina = 0
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        if slope == "Upsloping":
            slope = 1
        elif slope == "Flat":
            slope = 2
        else:
            slope = 3

        prediction1 = model1.predict(np.array([age, sex, cpt, bp, chol, fbp, ecg, mhr, exe_angina, oldpeak, slope]).reshape((1,-1)))
        output = round(prediction1[0])
        if output == 0:
            return render_template('result.html',prediction="Congratulations, you are not affected with heart disease. Have a good diet.!!")
        else:
            return render_template('result.html',prediction="Sorry ! But you have been affected with heart disease. Please consult doctor to avoid losses.")

    else:
        return render_template('heartcombined.html')

@app.route('/diabetespred', methods=['GET','POST'])
def diabetespred():
    if request.method == 'POST':
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = float(request.form['Glucose'])
        BloodPressure = float(request.form['BloodPressure'])
        Skinthickness = float(request.form['Skinthickness'])
        Insulin = float(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = int(request.form['Age'])
        
        prediction2 = model2.predict(np.array([Pregnancies, Glucose, BloodPressure, Skinthickness, Insulin, BMI, DiabetesPedigreeFunction, Age]).reshape((1,-1)))
        output = round(prediction2[0])
        if output == 0:
            return render_template('result.html',prediction="Congratulations, you are not affected with diabetes. Have a good diet.!!")
        else:
            return render_template('result.html',prediction="Sorry ! But you have been affected with diabetes. Please consult doctor to avoid losses.")

    else:
        return render_template('diabetes.html')

    

if __name__ == "__main__":
    app.run(debug=True)




