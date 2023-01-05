from datetime import time
import pickle

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
#Evaluation
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
import lightgbm as lgbm
from catboost import CatBoostClassifier
import lightgbm
from lightgbm import *
class Model1:
    """
    Model class to create various models and then ensembling it using custom techniques
    Written by : Akshada
    """

    def __init__(self,file_object,logger_object,X,Y):
        self.model_created = None
        self.X = X
        self.Y = Y
        self.file_object = file_object
        self.logger_object = logger_object
        self.logger_object.log(self.file_object, "Training model for diabetes dataset")

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
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.Y,test_size=0.20,stratify=Y)
            self.logger_object.log(self.file_object,"Data split into train and test")
            return self.X_train,self.X_test,self.y_train,self.y_test

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in train_test_split method of the Model class. Exception message:  ' + str(
                                       e))

            self.logger_object.log(self.file_object, "Train test split failed")

            raise Exception()

    def create_model(self, X_train, X_test, y_train, y_test):
        try:
            '''self.level0=list()
            self.level0.append(('gb',GradientBoostingClassifier(n_estimators=50,learning_rate=0.5)))
            self.level0.append(('lgbm', lightgbm.LGBMClassifier()))
            self.level0.append(('cart',DecisionTreeClassifier(random_state=5, ccp_alpha=0.0001, criterion='gini', min_samples_split=0.1,
                                              min_samples_leaf=1, splitter='best')))
            self.level0.append(('cat', CatBoostClassifier(learning_rate=0.2)))
            self.level1 = RandomForestClassifier(n_estimators=30, verbose=0)
            self.models = StackingClassifier(estimators=self.level0, final_estimator=self.level1, cv=10, n_jobs=-1)
            self.logger_object.log(self.file_object,"Fitting the training data")
            self.models.fit(self.X_train,self.y_train)
            #time.sleep(5)
            self.logger_object.log(self.file_object, "Custom Model created")

            self.filename = 'diabetes_prediction.pickle'
            pickle.dump(self.models, open(self.filename, 'wb'))
            self.logger_object.log(self.file_object, "Model saved in form of pickle")
            '''
            self.models=RandomForestClassifier()
            self.models.fit(self.X_train.values,self.y_train)
            self.logger_object.log(self.file_object, "Random forest Model created")

            self.filename = 'diabetes_prediction_rf.pickle'
            pickle.dump(self.models, open(self.filename, 'wb'))
            self.logger_object.log(self.file_object, "Model saved in form of pickle")
            return self.models

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


            self.logger_object.log(self.file_object,"Evaluation metrics of the model : ")

            y_pred = self.model_created.predict(X_test)


            self.logger_object.log(self.file_object,
                                   "Accuracy achived : " + str(accuracy_score(y_test,y_pred)))

            matrix = confusion_matrix(y_test,y_pred,labels=[1,0])
            #print('Confusion matrix : \n',matrix)
            self.logger_object.log(self.file_object,"Confusion matrix : \n" +str(matrix))

            # outcome values order in sklearn
            tp, fn, fp, tn = confusion_matrix(y_test, y_pred, labels=[1, 0]).reshape(-1)
            #print('Outcome values : \n', tp, fn, fp, tn)
            self.logger_object.log(self.file_object,'True positive : \n' +str(tp))
            self.logger_object.log(self.file_object, 'False negative : \n' + str(fn))
            self.logger_object.log(self.file_object, 'False positive : \n' + str(fp))
            self.logger_object.log(self.file_object, 'True negative : \n' + str(tn))

            # classification report for precision, recall f1-score and accuracy
            matrix = classification_report(y_test, y_pred, labels=[1, 0])
            #print('Classification report : \n', matrix)
            self.logger_object.log(self.file_object, 'Classification report : \n' + str(matrix))




        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured' + str(e))
            self.logger_object.log(self.file_object,
                                   'Model training failed. Exited run_tuner method of Models class')
            raise Exception()



