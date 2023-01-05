import pandas as pd
import numpy as np
from imblearn import over_sampling
from imblearn.over_sampling import RandomOverSampler

class DataPreprocessing1:
    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
    def replace_missing_value(self,data):
        self.logger_object.log(self.file_object, 'Entered the replace_missing_values method of the Preprocessor class')
        self.data = data
        try:
            self.data.loc[self.data['Glucose']==0, 'Glucose']= self.data['Glucose'].mean()
            self.data.loc[self.data['BMI'] == 0, 'BMI'] = self.data['BMI'].mean()
            self.data.loc[self.data['SkinThickness'] == 0, 'SkinThickness'] = self.data['SkinThickness'].mean()
            self.data.loc[self.data['Insulin'] == 0, 'Insulin'] = self.data['Insulin'].mean()
            self.data.loc[self.data['BloodPressure'] == 0, 'BloodPressure'] = self.data['BloodPressure'].mean()
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in replace_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Replacing missing values failed. Exited the replace_missing_values method of the Preprocessor class')
            raise Exception()
    def remove_outliers(self,data):
        Q1=data.quantile(0.25)
        Q3=data.quantile(0.75)
        IQR=Q3-Q1
        df_out = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
        return df_out
    def balance_data(self,data):

        """
        Method to balance data using RandomOverSampler
        input:
        :param X:
        :param Y:
        Output : Balanced dataframe with equal number of labels
        on failure : Raise exception
        """
        try:
            X = data.drop(["Outcome"], axis=1)
            y = data["Outcome"]
            ros = RandomOverSampler(random_state=0)
            X_resampled, y_resampled = ros.fit_resample(X, y)

            return X_resampled,y_resampled

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in balance_data method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Data Balancing Unsuccessful. Exited the balance_data method of the Preprocessor class')
            raise Exception()



    def run_preprocess(self,data):

        data1=self.replace_missing_value(data)
        #X, Y = self.seperate_labels_features(data, 'Outcome')
        #self.logger_object.log(self.file_object, "Labels and features seperated")
        data2=self.remove_outliers(data1)
        X, Y = self.balance_data(data2)
        self.logger_object.log(self.file_object, "Data Balanced")
        print(Y.value_counts())
        print(X.isnull().sum())
        return X,Y



