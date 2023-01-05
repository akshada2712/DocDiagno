import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import KNNImputer

class DataPreprocessing:
    """
    Class to prepare the data for training and creating the model
    Written by : Akshada
    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def drop_feature(self,data,ft):
        """
        Method to drop a particular feature
        input:
        :param data:
        :param ft -> feature to be dropped:
        :return:
        output : data -> whose unused feature is removed
        On failure : Raise Exception
        """
        try:
            self.data = data.drop([ft],axis=1,inplace=True)
            self.logger_object.log(self.file_object,"Feature dropped")
            return self.data

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in drop_feature method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Dropping feature Unsuccessful. Exited the drop_feature method of the Preprocessor class')
            raise Exception()


    def seperate_labels_features(self,data,label_name):

        """
        Method to seperate features and labels
        input:
        :param data:
        :param label_name:
        output: X,Y (X-> features, Y -> labels)
        On failure : Raise exception
        """

        self.logger_object.log(self.file_object,"Entered the datapreprocessing")
        try:
            self.X = data.drop(labels=label_name, axis=1)
            self.Y = data[label_name]
            self.logger_object.log(self.file_object, "Features and labels seperated")
            return self.X, self.Y

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def is_null_present(self,data):

        """
        Method to check null count
        input:
        :param data:
        output: dataframe with null count
        On failure : Raise exception
        """

        self.logger_object.log(self.file_object,"Entered checking null values count")
        self.null_present = False
        try:
            self.null_counts = data.isna().sum()  # check for the count of null values per column
            for i in self.null_counts:
                if i > 0:
                    self.null_present = True
                    break
            if self.null_present:  # write the logs to see which columns have null values
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                dataframe_with_null.to_csv("Combine_Null.csv")
                return self.null_present
        except Exception as e:
            self.logger.exception(
                'Exception occurred in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            raise Exception()


    def check_std_deviation(self,data):

        """
        Method to check standard deviation. If zero drop that column
        input:
        :param data:
        output: data frame with columns dropped whose std = 0
        on failure : Raise exceptions
        """
        self.logger_object.log(self.file_object,
                               'Entered the get_columns_with_zero_std_deviation method of the Preprocessor class')
        self.columns = data.columns
        self.data_n = data.describe()
        self.col_to_drop = []
        try:
            for x in self.columns:
                if (self.data_n[x]['std'] == 0):  # check if standard deviation is zero
                    self.col_to_drop.append(x)  # prepare the list of columns with standard deviation zero
            self.logger_object.log(self.file_object,
                                   'Column search for Standard Deviation of Zero Successful. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            for i in self.col_to_drop:
                self.data = data.drop(i,axis=1,inplace=True)

            return self.col_to_drop

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_columns_with_zero_std_deviation method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Column search for Standard Deviation of Zero Failed. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            raise Exception()

    def replace_mode(self,col_name):
        """
        Method to replace null values using the mode of that column
        input:
        :param col_name -> col with null values to be replaced with mode:
        output : null values of col removed
        on failure : Raise exception
        """
        try:
            self.mode = self.X[col_name]
            self.X[col_name].fillna(self.mode[0],inplace=True)
            self.logger_object.log(self.file_object,"Null value replaced with mode")
            return self.X

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in replace_mode method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'replace_mode Unsuccessful. Exited the replace_mode method of the Preprocessor class')
            raise Exception()


    def impute_missing_values(self, data):
        """
                                        Method Name: impute_missing_values
                                        Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
                                        Output: A Dataframe which has all the missing values imputed.
                                        On Failure: Raise Exception


                     """
        self.logger_object.log(self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')
        self.data= data
        try:
            imputer=KNNImputer(n_neighbors=3, weights='uniform',missing_values=np.nan)
            self.new_array=imputer.fit_transform(self.data) # impute the missing values
            # convert the nd-array returned in the step above to a Dataframe
            self.new_data=pd.DataFrame(data=self.new_array, columns=self.data.columns)
            self.logger_object.log(self.file_object, 'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return self.new_data

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()






    def balance_data(self,X,Y):

        """
        Method to balance data using RandomOverSampler
        input:
        :param X:
        :param Y:
        Output : Balanced dataframe with equal number of labels
        on failure : Raise exception
        """
        try:
            self.os = RandomOverSampler(random_state=42)
            self.X,self.Y = self.os.fit_sample(X,Y)
            self.logger_object.log(self.file_object, "Data Balanced")

            return self.X,self.Y

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in balance_data method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Data Balancing Unsuccessful. Exited the balance_data method of the Preprocessor class')
            raise Exception()


    def run_preprocessing(self,data):
        """
        method to run all preprocessing steps
        :param data:
        output : processed labels and features
        On failure: Raise Exception
        """
        try:
            self.logger_object.log(self.file_object,'Running Preproccessing')

            X,Y=self.seperate_labels_features(data,'target')
            self.logger_object.log(self.file_object,"Labels and features seperated")
            #print(X.shape,Y.shape)

            self.is_null_present(X)
            self.logger_object.log(self.file_object,"Null values checked")

            self.check_std_deviation(X)
            self.logger_object.log(self.file_object, "Std deviation checked")

            print(Y.value_counts())
            X,Y = self.balance_data(X,Y)
            self.logger_object.log(self.file_object, "Data Balanced")
            print(Y.value_counts())

            return X,Y

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in run_preprocessing method of the Preprocessor class. Exception message:  ' + str(
                                       e))

            self.logger_object.log(self.file_object,"Preprocessing failed.")

            raise Exception()





