import numpy as np
import re

class TypeInferer(object):
    """Abstract base class for all type inference modules.
    """
    self.digit_sep = ","
    def infer(self, array):
        raise NotImplementedError()

class RegexTypeInferer(TypeInferer):
    """A regex based type inference module.
    """
    def infer(self, array):
        

    def __test_disc_num(array):
        int_pattern = re.compile("[0-9]+$")

    def __test_cont_num(array):
        float_pattern = re.compile("[0-9]*[.][0-9]+$|[0-9]+[.][0-9]*$")

    def __test_date(array):
        date_pattern = re.compile("[0-9]{2,2}[-][0-9]{2,2}[.|/|-][0-9]{2,4}$|[0-9]{2,4}[.|/|-][0-9]{2,2}[.|/|-][0-9]{2,2}$")

    # sub-tests for cont:
    def __test_cont_interval(array):
        pass

    def __test_real_pos(array):
        pass

    # sub-test for disc:
    def __test_count(array):
        pass


    
