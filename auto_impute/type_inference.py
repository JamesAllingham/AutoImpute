import numpy as np
import re
from DataType import DataType

class TypeDeducer(object):
    """Abstract base class for all type inference modules.
    """
    self.digit_sep = ","
    def defer(self, array):
        raise NotImplementedError("This method must be implemented by all child classes.")

class RegexTypeDeducer(TypeDeducer):
    """A regex based type inference module.
    """
    def deduce(self, array):
        if __test_real_num(array):
            return DataType.Real
        elif __test_integer_num(array):
            return DataType.Integer
        elif __test_date(array):
            return DataType.Date
        else:
            return DataType.Unknown

    def __do_regex_test(array, pattern):
        passed_test = True
        for item in array:
            passed_test = False if not pattern.match(item) else passed_test
        return passed_test

    def __test_integer_num(array):
        int_pattern = re.compile("[0-9]+$")
        return __do_regex_test(array, int_pattern)

    def __test_real_num(array):
        float_pattern = re.compile("[0-9]*[.][0-9]+$|[0-9]+[.][0-9]*$")
        return __do_regex_test(array, float_pattern)

    def __test_date(array):
        date_pattern = re.compile("[0-9]{2,2}[-][0-9]{2,2}[.|/|-][0-9]{2,4}$|[0-9]{2,4}[.|/|-][0-9]{2,2}[.|/|-][0-9]{2,2}$")
        return __do_regex_test(array, date_pattern)

    # # sub-tests for cont:
    # def __test_cont_interval(array):
    #     pass

    # def __test_real_pos(array):
    #     pass

    # # sub-test for disc:
    # def __test_count(array):
    #     pass   
