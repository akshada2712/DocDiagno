from application_logging.logger import App_Logger
from DataLoading.data_loader import DataLoading
from DataPreprocessing.preprocessing import DataPreprocessing
from Model_training.tuner import Model
from Predictions.predict_combine import MakePredictions
from Predictions.predict_framingham import MakePredictionsFramingham
from Framingham_training.framingham_tuner import Model_framingham


"""
Main function to run all classes. Need to modify while creating webapp
Written by : Akshada"""

# creating file and logger object
logger = App_Logger()
file = open('Logs/Application.txt','a+')
message = "Application started"
logger.log(file,message)

# loading the dataset
#data_file = 'combine_heart.csv'
"""data_file = 'framingham.csv'
load = DataLoading(file,logger,data_file)
data = load.get_data()

# preprocessing the dataset
data_preprocessor = DataPreprocessing(file,logger)
X,Y = data_preprocessor.run_preprocessing(data)

#creating and training the model
model_train = Model(file,logger,X,Y)
model_train.run_tuner()

# making predictions
filename = "combine_heart.pickle"
predictions = MakePredictions(file,logger,filename)
predictions.runPredictions()"""

ans = input("Do you have diabetes [y/n]:")

#data = 'fram'
#data = 'mmm'
if ans == 'y':
    print("In framingham")
    #loading the dataset

    data_file = 'framingham.csv'
    load = DataLoading(file,logger,data_file)
    data = load.get_data()
    #print(data)

    # preprocessing data
    data_preprocessor = DataPreprocessing(file,logger)
    ft = 'education'
    data_preprocessor.drop_feature(data,ft)
    #print(data)
    label = 'TenYearCHD'

    X, Y = data_preprocessor.seperate_labels_features(data,label)
    data_preprocessor.is_null_present(X)
    data_preprocessor.check_std_deviation(X)

    col_name = 'BPMeds'
    X = data_preprocessor.impute_missing_values(data_preprocessor.replace_mode(col_name))
    X,Y = data_preprocessor.balance_data(X,Y)
    #print(Y.value_counts())

    # creating and training the model
    model_train = Model_framingham(file, logger, X, Y)
    y_pred = model_train.run_tuner()
    #print(y_pred)

    """
    Web application work starts here
    """

    gender = int(input("What is your gender : (1-> male; 0-> female) "))
    age = int(input("What is your age : "))
    smoker = int(input("Do you smoke : (0 -> no; 1->yes) "))
    cigs = int(input("How many cigarettes do you consume : "))
    bp_meds = int(input('Do you take bp medicines : (0 -> no; 1->yes) '))
    stroke = int(input('Do you suffer from stroke : (0 -> no; 1->yes) '))
    hyp = int(input("Do you suffer from hypertension : (0 -> no; 1->yes) "))
    dia = 1
    chol = int(input('How much is your cholestrol : '))
    sysBp = int(input("What is your systolic bp : "))
    diaBp = int(input("What is your diastolic bp : "))
    height = float(input('Please enter your height input meters(decimals): '))
    weight = int(input('Please enter your weight input kg: '))
    bmi = weight / (height * height)
    rate = int(input('What is your pulse/heart rate : '))
    glu = float(input('What is your sugar : '))

    # making predictions
    filename = "framingham.pickle"
    predictions = MakePredictionsFramingham(file, logger, filename)
    loaded_model = predictions.load_model()
    predictions.prediction_app(loaded_model,gender,age,smoker,cigs,bp_meds,stroke,hyp,dia,chol,sysBp,diaBp,bmi,rate,glu)
    #predictions.runPredictions()


else:
    print("In combined")
    # loading the dataset
    data_file = 'combine_heart.csv'
    load = DataLoading(file,logger,data_file)
    data = load.get_data()

    #preprocessing data
    data_preprocessor = DataPreprocessing(file, logger)
    label = 'target'
    X, Y = data_preprocessor.seperate_labels_features(data, label)
    data_preprocessor.is_null_present(X)
    data_preprocessor.check_std_deviation(X)

    X, Y = data_preprocessor.balance_data(X, Y)
    #print(Y.value_counts())

    # creating and training the model
    model = Model(file,logger,X,Y)
    model.run_tuner()


    """
    Web application work starts here
    """
    age = int(input('What is your age : '))
    sex = int(input('Whats your gender : (1 -> Male, 0 -> Female) '))
    cpt = int(input('What is chest pain type : (1 - typical angina; 2 - atypical angina; 3 - nominal angina; 4 - asymptomatic angina) '))
    bp = int(input('What is your blood pressure : '))
    chol = int(input('How much is your cholestrol : '))
    fbp = int(input('What is your fasting sugar : '))
    if fbp >= 120 :
        fbp = 1
    else :
        fbp = 0

    ecg = int(input('Ecg status : (resting ecg - > 0 : normal, 1 - ST-T wave abnormality; 2 - left ventricular hypertrophy) '))
    #max heart rate
    mhr = 220 - age
    exe_angina = 0
    if cpt == 4 :
        exe_angina = 1
    else:
        exe_angina = 0
    oldpeak = float(input("What is your oldpeak score : "))
    slope = int(input("What is the st slope of your ecg : (1 = upsloping, 2 - flat, 3 - downsloping) "))



    # making predictions
    filename = 'combine_heart.pickle'
    predictions = MakePredictions(file,logger,filename)
    loaded_model = predictions.load_model()
    predictions.prediction_app(loaded_model,age,sex,cpt,bp,chol,fbp,ecg,mhr,exe_angina,oldpeak,slope)
    #predictions.runPredictions()





