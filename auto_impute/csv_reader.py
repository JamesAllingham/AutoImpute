# John Doe
# Feb 2018
# csv_reader.py
# CSV reader for AutoImpute

import numpy as np
import numpy.ma as ma
import csv

class CSVReader(object):
    """ Manages the data read from a CSV file.
    Can return the raw data as well as masks indicating missing values and labels for column names.
    """

    def __init__(self, file_name, delimiter=',', header=True, indicator=''):
        
        self.__raw_data = []
        self.__header = None
        with open(file_name, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            for i, row in enumerate(reader):
                if i == 0 and header:
                    self.__header = row
                    continue
                
                self.__raw_data.append(row)

        # create a masked array to store the data
        N = len(self.__raw_data)
        num_features = len(self.__raw_data[0])
        data = np.zeros(shape=(N, num_features))
        mask = np.zeros_like(data)

        for i in range(N):
            for j in range(num_features):
                if self.__raw_data[i][j] == indicator:
                    mask[i,j] = True
                else:
                    data[i,j] = float(self.__raw_data[i][j])

        self.__masked_data = ma.masked_array(data=data, mask=mask)

    def get_raw_data(self):
        return self.__raw_data

    def get_masked_data(self):
        return self.__masked_data

    def get_column_names(self):
        if self.__header:
            return self.__header        
        else:
            raise ValueError('No column names were inferred.')
