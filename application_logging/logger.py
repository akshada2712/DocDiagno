from datetime import datetime

class App_Logger:
    """
    Class to log different events
    input: file, logger_object
    output : message logged in given file
    """
    def __init__(self):
        pass

    def log(self, file_object, log_message):
        """
        method to log message
        :param file_object:
        :param log_message:
        :return:
        """
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        file_object.write(
            str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message +"\n")
