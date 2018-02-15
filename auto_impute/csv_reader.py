import numpy as np

class CSVReader(object):
    """ Manages the data read from a CSV file.
    Can return the raw data as well as masks indicating missing values and labels for column names.
    """

    def __init__(file_name, delimiter=',', infer_header=True)
        self.raw_data = np.genfromtxt(dtype="str", delimiter=delimiter)
        
        if infer_header:
            self.header = raw_data[0]
            self.columns = raw_data[1:].T
        else:
            self.header = None
            self.columns = raw_data.T        

    def get_raw_data():
        return self.raw_data

    def get_column_names():
        if self.header:
            return self.header        
        else:
            raise ValueError('No column names were inferred.')

    def get_column(col_idx):
        if col_idx > columns.shape[0]:
            raise ValueError('Column index greater than number of columns')
        return columns[col_idx]
