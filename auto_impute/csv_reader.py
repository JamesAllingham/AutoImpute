# James Allingham
# Feb 2018
# csv_reader.py
# CSV reader for AutoImpute

import numpy as np
import pandas as pd

class CSVReader(object):
    """ Manages the data read from a CSV file.
    Can return the raw data as well as masks indicating missing values and labels for column names.
    """

    def __init__(self, file_name, delimiter=',', infer_header=True):
        self.__raw_data = np.genfromtxt(file_name, delimiter=delimiter)
        
        if infer_header:
            self.__header = self.__raw_data[0]
            self.__columns = self.__raw_data[1:].T
        else:
            self.__header = None
            self.__columns = self.__raw_data.T        

    def get_raw_data(self):
        return self.__raw_data

    def get_column_names(self):
        if self.__header:
            return self.__header        
        else:
            raise ValueError('No column names were inferred.')

    def get_column(self, col_idx):
        if col_idx > self.__columns.shape[0]:
            raise ValueError('Column index greater than number of columns')
        return self.__columns[col_idx]
