from application_logging.logger import App_Logger1
from DataLoading.data_loader import DataLoading1
from DataPreprocessing.preprocessing import DataPreprocessing1
from Model_training.tuner import Model1
from Predictions.predict_combine import MakePredictions1


logger1 = App_Logger1()
file1 = open('logs_diabetes/diabetes_logs.txt','a+')
message = "Application started"
logger1.log(file1,message)
print("---------------------In diabetes-----------------")
data_file1 = 'Diabetes/diabetes.csv'
load = DataLoading1(file1,logger1,data_file1)
data = load.get_data()

#preprocessinh
data_preprocessor = DataPreprocessing1(file1, logger1)
X,Y=data_preprocessor.run_preprocess(data)
#-----training model
model_diabetes = Model1(file1,logger1,X,Y)
model_diabetes.run_tuner()
#------input for predictions-------
Pregnancies=int(input("Number of times u were pregnant"))
Glucose=float(input("Glucose concentration level a 2 hours in an oral glucose tolerance test "))
BP=float(input("Blood Pressure"))
SkinThickness=float(input("Triceps skin fold thickness in mm"))
Insulin=float(input("Insulin "))
BMI=float(input('BMI'))
DPF=float(input('Diabetes pedigree Function'))
Age=int(input("age"))
#----------making prediction--------

filename1 = 'diabetes_prediction_rf.pickle'
predictions = MakePredictions1(file1,logger1,filename1)
loaded_model = predictions.load_model()
predictions.prediction_app(loaded_model,Pregnancies,Glucose,BP,SkinThickness,Insulin,BMI,DPF,Age)