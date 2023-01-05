import pandas as pd
import numpy as np
import pickle

class MakePredictions1:

    """
    Class for making predictions
    Written by : Akshada
    """

    def __init__(self,file_object,logger_object,filename):
        """
        Initiating the instance of class.
        Input: filename -> name of model file(xyz.pickle)"""
        self.filename = filename
        self.file_object = file_object
        self.logger_object = logger_object

    def load_model(self):

        """
        Method for loading pickle model file for making predictions
        On failure : Raise exception
        :return:
        """
        try:
            loaded_model = pickle.load(open(self.filename, 'rb'))
            self.logger_object.log(self.file_object,"Model is loaded for predictions")
            return loaded_model

        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured' + str(e))
            self.logger_object.log(self.file_object,
                                   'Pickle file loading unsuccesfull .Exited the load_model method of the MakePredictions class')
            raise Exception()

    def prediction_app(self,model,Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
        """
        Method for making predictions.
        Input:
        :param model:
        parameters or features
        :return:
        Output: class predicted
        on failure : Raise Exception
        """
        try:


            self.model = model

            a = self.model.predict(np.array(
                [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,
                 Age]).reshape(1,-1))

            self.logger_object.log(self.file_object, "Predicted output" + str(a))

            if a[0] == 0:
                self.logger_object.log(self.file_object,"Congratulations, you are not affected with Diabetes. Have a good diet.!!")
                #print("Congratulations, you are not affected with heart disease. Have a good diet.!!")

            else:
                self.logger_object.log(self.file_object,"Sorry ! But you have been affected with Diabetes. Please consult doctore to avoid losses")
                #print("Sorry ! But you have been affected with heart disease. Please consult doctore to avoid losses")

        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured' + str(e))
            self.logger_object.log(self.file_object,
                                   'Predictions making unsuccessfull.Exited the prediction_app method of the MakePredictions class')
            raise Exception()

    def runPredictions(self):
        """
        Method to run all functions of MakePrediction class
        :return:
        """
        try:
            Load_model = self.load_model()

            self.prediction_app(Load_model)

        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured' + str(e))
            self.logger_object.log(self.file_object,
                                   'running predictions Unsuccessful.Exited the runPredictions method of the MakePredictionsClass')
            raise Exception()