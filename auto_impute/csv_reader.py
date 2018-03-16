# James Allingham
# Feb 2018
# csv_reader.py
# CSV reader for AutoImpute

import numpy as np

class CSVReader(object):
    """ Manages the data read from a CSV file.
    Can return the raw data as well as masks indicating missing values and labels for column names.
    """

    def __init__(file_name, delimiter=',', infer_header=True)
        self.__raw_data = np.genfromtxt(delimiter=delimiter)
        
        if infer_header:
            self.__header = raw_data[0]
            self.__columns = raw_data[1:].T
        else:
            self.__header = None
            self.__columns = raw_data.T        

    def get_raw_data():
        return self.__raw_data

    def get_column_names():
        if self.__header:
            return self.__header        
        else:
            raise ValueError('No column names were inferred.')

    def get_column(col_idx):
        if col_idx > self.__columns.shape[0]:
            raise ValueError('Column index greater than number of columns')
        return self.__columns[col_idx]
