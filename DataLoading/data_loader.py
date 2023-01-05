import pandas as pd
from application_logging.logger import App_Logger

class DataLoading:

    """
    class for loading the data
    Written by : Akshada
    """
    def __init__(self, file_object,logger_object,data):
        self.data = data
        self.file_object = file_object
        self.logger_object = logger_object


    def get_data(self):
        """
        method to get_data from file
        input: data_file
        Output : Dataframe
        On Failure : Raise exception

        """
        try:
            self.data = pd.read_csv(self.data)
            self.logger_object.log(self.file_object,"Data loaded successfully")
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured' + str(e))
            self.logger_object.log(self.file_object,
                                   'Data Load Unsuccessful.Exited the get_data method of the DataLoading class')
            raise Exception()


