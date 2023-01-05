import pandas as pd
import pickle

import xgboost
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time

class Model_framingham:
    """
    Model class to create various models and then ensembling it using custom techniques
    Written by : Akshada
    """

    def __init__(self,file_object,logger_object,X,Y):
        self.X = X
        self.Y = Y
        self.file_object = file_object
        self.logger_object = logger_object

        self.logger_object.log(self.file_object,"Training model for framingham dataset")

    def split_train_test(self,X,Y):

        """
        Method to split data in training and test sets
        input:
        :param X:
        :param Y:
        output : X_train,X_test,y_train,y_test
        On failure : Raise Exception
        """
        try:
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.Y,test_size=0.20,random_state=355,stratify=self.Y)
            self.logger_object.log(self.file_object,"Data split into train and test")
            return self.X_train,self.X_test,self.y_train,self.y_test

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in train_test_split method of the Model class. Exception message:  ' + str(
                                       e))

            self.logger_object.log(self.file_object, "Train test split failed")

            raise Exception()


    def create_model(self,X_train,X_test,y_train,y_test):
        """
        Method to create different models and then using voting classifier to get the best accuracy
        The class with maximum votes(i.e predictions) from different models will be the result
        input:
        :param X_train:
        :param X_test:
        :param y_train:
        :param y_test:
        Output:
        Model saved in form of pickle file
        On failure : Raise Exception
        """

        try:

            self.models = list()

            self.dtree =  Pipeline([('m',DecisionTreeClassifier())])
            self.models.append(('decisionTree',self.dtree))
            self.logger_object.log(self.file_object, "Decision tree model added")

            self.rfm = Pipeline([('m',RandomForestClassifier())])
            self.models.append(('randomForest',self.rfm))
            self.logger_object.log(self.file_object, "Random Forest Classifier model added")

            self.xgb = Pipeline([('m',XGBClassifier())])
            self.models.append(('xgboost',self.xgb))
            self.logger_object.log(self.file_object, "XGBoost model added")

        # define the voting ensemble
            self.ensemble = VotingClassifier(estimators=self.models, voting='hard')
            self.logger_object.log(self.file_object,"Fitting the training data")
            self.ensemble.fit(self.X_train, self.y_train)
            time.sleep(5)
            self.logger_object.log(self.file_object, "Custom Model created")
            #self.y_pred = self.ensemble.predict(self.X_test)
            #self.logger_object.log(self.file_object,"Accuracy achived : " +str(accuracy_score(self.y_test,self.y_pred)))

            self.filename = 'framingham.pickle'
            pickle.dump(self.ensemble, open(self.filename, 'wb'))
            self.logger_object.log(self.file_object, "Model saved in form of pickle")

            return self.ensemble

        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured' + str(e))
            self.logger_object.log(self.file_object,
                                   'Creating model failed. Exited the create_model method of Model class')
            raise Exception()



    def run_tuner(self):
        """
        Method to run model tuner
        output: Model created and saved
        on failure : Raise Exception
        """
        try:
            X_train, X_test, y_train, y_test = self.split_train_test(self.X,self.Y)

            self.model_created = self.create_model(X_train, X_test, y_train, y_test)

            self.logger_object.log(self.file_object, "Evaluation metrics of the model : ")

            y_pred = self.model_created.predict(X_test)

            self.logger_object.log(self.file_object,
                                   "Accuracy achived : " + str(accuracy_score(y_test, y_pred)))

            matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
            # print('Confusion matrix : \n',matrix)
            self.logger_object.log(self.file_object, "Confusion matrix : \n" + str(matrix))

            # outcome values order in sklearn
            tp, fn, fp, tn = confusion_matrix(y_test, y_pred, labels=[1, 0]).reshape(-1)
            # print('Outcome values : \n', tp, fn, fp, tn)
            self.logger_object.log(self.file_object, 'True positive : \n' + str(tp))
            self.logger_object.log(self.file_object, 'False negative : \n' + str(fn))
            self.logger_object.log(self.file_object, 'False positive : \n' + str(fp))
            self.logger_object.log(self.file_object, 'True negative : \n' + str(tn))

            # classification report for precision, recall f1-score and accuracy
            matrix = classification_report(y_test, y_pred, labels=[1, 0])
            # print('Classification report : \n', matrix)
            self.logger_object.log(self.file_object, 'Classification report : \n' + str(matrix))


        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured' + str(e))
            self.logger_object.log(self.file_object,
                                   'Model training failed. Exited run_tuner method of Models class')
            raise Exception()



        #self.save_pickle(self.model_created)