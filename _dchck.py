# check.py

# Module import
from inspect import *
import pandas as pd
import numpy as np
import sys
import pathlib2
import os.path
import periodic


__doc__ = "\n-------------------------------------------------------------------------\n" \
          "This module provides functions for error checking to use either for other\n" \
          "df_tools modules or otherwise.\n" \
          "\nFunctions:\n" \
          "check_ls, check_eq_ls_len, check_numeric, check_int, check_string,\n" \
          "check_bool, check_dfs, param_exists_in_set, check_threshold,\n" \
          "parent_fn_mod_2step, parent_fn_mod_3step.\n\n" \
          "Please see the doc strings of individual functions for further information.\n" \
          "-------------------------------------------------------------------------\n"


# 1. Object Type/Value and List length checking
def check_ls(ls, nt_flag=False):
    """
    This function checks if an object is a list. If not, it returns an error, noting that a list is required.

    Parameters:
    :param ls: (any object) Object to check. A list is expected.
    :param nt_flag: (boolean) Non-termination flag. That is, if the condition is not met and nt_flag==True, then the script is
        not terminated. If nt_flag==False, then the script is terminated. This allows the user to use multiple checks
        as a condition without terminating the script if one of them is met.
    :return: err_flag: (boolean) The result of the error check. True means that the object passed to [ls] is not a list.
        False means that it was.
    """
    param_exists_in_set(value=nt_flag, val_set=[True, False])
    if isinstance(ls, list):
        err_flag = False
        pass
    else:
        err_flag = True
        main_module, main_fn, main_lineno = parent_fn_mod_3step()
        calling_module, calling_fn, calling_lineno = parent_fn_mod_2step()
        if not nt_flag:
            print('On line %i in function %s of module %s' % (main_lineno, main_fn, main_module))
            print('     Error on line %i in module %s' % (calling_lineno, calling_module))
            print('         Invalid input for function %s' % calling_fn)
            sys.exit('          ERROR: A list is required.')
        else:
            pass
    return err_flag


def check_np_array(arr, nt_flag=False):
    """
    This function checks if an object is a numpy.array. If not, it returns an error, noting that a list is required.

    Parameters:
    :param arr: (any object) Object to be checked. A numpy array is expected.
    :param nt_flag: (boolean) Non-termination flag. That is, if the condition is not met and nt_flag==True, then the script is
        not terminated. If nt_flag==False, then the script is terminated. This allows the user to use multiple checks
        as a condition without terminating the script if one of them is met.
    :return: err_flag: (boolean) The result of the error check. True means that the object passed to [arr] is not an
        np array. False means that it was.
    """

    if isinstance(arr, np.ndarray):
        err_flag = False
        pass
    else:
        err_flag = True
        main_module, main_fn, main_lineno = parent_fn_mod_3step()
        calling_module, calling_fn, calling_lineno = parent_fn_mod_2step()
        if not nt_flag:
            print('On line %i in function %s of module %s' % (main_lineno, main_fn, main_module))
            print('     Error on line %i in module %s' % (calling_lineno, calling_module))
            print('         Invalid input for function %s' % calling_fn)
            sys.exit('          ERROR: A list is required.')
        else:
            pass

    return err_flag


def check_eq_ls_len(list_ls=[]):
    """
    This function checks a list of lists to ensure that they are all the same length. If they are not, an error is
        returned, and the script is exited.

    Parameters:
    :param list_ls: (any object) A list of lists in which each member's length will be compared to the others.
    :return: 0
    """
    for ls_no in range(0, len(list_ls) - 1):
        if len(list_ls[ls_no]) == len(list_ls[ls_no + 1]):
            pass
        else:
            main_module, main_fn, main_lineno = parent_fn_mod_3step()
            calling_module, calling_fn, calling_lineno = parent_fn_mod_2step()
            print('On line %i in function %s of module %s' % (main_lineno, main_fn, main_module))
            print('     Error on line %i in module %s' % (calling_lineno, calling_module))
            print('         Invalid input for function %s' % calling_fn)
            sys.exit('          ERROR: Length of at least 2 lists are unequal.')

    return 0


def check_num_equal(val1, val2):
    """
    Function to check if two values are equal. Expects a numerical argument
    :param val1: Numerical value 1 to check
    :param val2: Numerical value 2 to check
    :return: Nothing if check is passed. Otherwise, returns error to terminal and exits script
    """
    if val1 == val2:
        pass
    else:
        main_module, main_fn, main_lineno = parent_fn_mod_3step()
        calling_module, calling_fn, calling_lineno = parent_fn_mod_2step()
        print('On line %i in function %s of module %s' % (main_lineno, main_fn, main_module))
        print('     Error on line %i in module %s' % (calling_lineno, calling_module))
        print('         Invalid input for function %s' % calling_fn)
        sys.exit('          ERROR: Two values are not equal.')

    return 0


def check_numeric(values=[]):
    """
    This function checks to see if all values in a list are numerical. It passed tests with nans and strings. If a
        non-numerical value is found (by type cast to integer), then it an error is returned and the script stops. The
        resulting error will indicate the calling module, function and line number within the calling module.

    Parameters:
    :param values: (any object) A list of values to check to ensure that they are numeric. If they are not, an error is returned,
        and the script is exited.
    :return: 0
    """
    # Todo: The following numpy types were not being recognized at a certain point during testing. The cause is
    # todo: unknown since they were initially working fine. Will work to re-implement them in future versions.
    # Problematic types: np.int128, no.float80, np.float96, np.float128, np.float256, np.uint128, np.int128

    numeric_types = (int, float, complex, np.int, np.int8, np.int16, np.int32, np.int64, np.float,
                     np.float16, np.float32, np.float64, np.uint8, np.uint16, np.uint32, np.uint64)
    for val, valnum in zip(values, range(len(values))):
        if isinstance(val, numeric_types):
            pass
        else:
            main_module, main_fn, main_lineno = parent_fn_mod_3step()
            calling_module, calling_fn, calling_lineno = parent_fn_mod_2step()
            print('On line %i in function %s of module %s' % (main_lineno, main_fn, main_module))
            print('     Error on line %i in module %s' % (calling_lineno, calling_module))
            print('         Non-numeric value "%s" found list at index %i!' % (val, valnum))
            sys.exit('         ERROR: All values must be of numerical type (int, float, complex, numpy.*.')
    return 0


def check_int(values=[]):
    """
    This function checks if every item in a list is an integer. Otherwise, it returns an error and exits the script.

    Parameters:
    :param values: (any object) List of values/objects that will be tested as to whether or not they are integers. If any are not,
        an error is returned, and the script is exited.
    :return: 0
    """
    for val, valnum in zip(values, range(len(values))):
        if isinstance(val, int):
            pass
        else:
            main_module, main_fn, main_lineno = parent_fn_mod_3step()
            calling_module, calling_fn, calling_lineno = parent_fn_mod_2step()
            print('On line %i in function %s of module %s' % (main_lineno, main_fn, main_module))
            print('     Error on line %i in module %s' % (calling_lineno, calling_module))
            print('         Non-integer object "%s" found list at index %i!' % (val, valnum))
            sys.exit('         ERROR: All values must be integers.')
    return 0


def check_string(values=[]):
    """
    This function checks if every item in a list is an integer. Otherwise, it returns an error and exits the script.

    Parameters:
    :param values: (any object) List of values/objects that will be tested as to whether or not they are strings. If any are not,
        an error is returned and the script is exited.
    :return: 0
    """
    for val, valnum in zip(values, range(len(values))):
        if isinstance(val, str):
            pass
        else:
            main_module, main_fn, main_lineno = parent_fn_mod_3step()
            calling_module, calling_fn, calling_lineno = parent_fn_mod_2step()
            print('On line %i in function %s of module %s' % (main_lineno, main_fn, main_module))
            print('     Error on line %i in module %s' % (calling_lineno, calling_module))
            print('         Non-string frame object "%s" found list at index %i!' % (val, valnum))
            sys.exit('         ERROR: All values must be strings.')
    return 0


def check_bool(values=[]):
    """
    This function checks whether or not a list of values are boolean. If not, returns an error, echos to the terminal
        and exits the script.

    Parameters:
    :param values: (any object) Values to check whether or not they are boolean (i.e., True/False)
    :return: 0
    """
    for val, valnum in zip(values, range(len(values))):
        if isinstance(val, bool):
            pass
        else:
            main_module, main_fn, main_lineno = parent_fn_mod_3step()
            calling_module, calling_fn, calling_lineno = parent_fn_mod_2step()
            print('On line %i in function %s of module %s' % (main_lineno, main_fn, main_module))
            print('     Error on line %i in module %s' % (calling_lineno, calling_module))
            print('         Non-bool object "%s" found list at index %i!' % (val, valnum))
            sys.exit('         ERROR: All values must be boolean (True/False).')

    return 0


def check_dfs(values=[]):
    """
    This function checks if every item in a list is a pandas DataFrame. If any are not, an error is returned, and the
        script is exited.

    Parameters:
    :param values: (any object) List of values/objects that will be tested as to whether or not they are pandas DataFrames.
    :return: 0
    """
    for val, valnum in zip(values, range(len(values))):
        if isinstance(val, pd.DataFrame):
            pass
        else:
            main_module, main_fn, main_lineno = parent_fn_mod_3step()
            calling_module, calling_fn, calling_lineno = parent_fn_mod_2step()
            print('On line %i in function %s of module %s' % (main_lineno, main_fn, main_module))
            print('     Error on line %i in module %s' % (calling_lineno, calling_module))
            print('         Non-DataFrame object "%s" found in list at index %i!' % (val, valnum))
            sys.exit('         ERROR: All values must be pandas DataFrames.')
    return 0


def check_pathlib_path(values=[]):
    """
    This function checks whether or not a value is a legitimate pathlib2 path type (pathlib.Path)
    :param values:
    :return: path_check: (boolean) Result of check for pathlib2.Path type for [values]. If true, all paths exist. If
        False, at least one does not.
    """
    path_check = False
    for item in values:
        if isinstance(item, pathlib2.Path):
            path_check = True
        else:
            path_check = False

    return path_check


def param_exists_in_set(value, val_set=[]):
    """
    This function checks if the passed value exists in a set of values
    :param value: (any object) The value to check for in the set.
    :param val_set: (list of any type) Set of values in which to check for parameter, value
    :return: 0
    """
    if value in val_set:
        pass
    else:
        main_module, main_fn, main_lineno = parent_fn_mod_3step()
        calling_module, calling_fn, calling_lineno = parent_fn_mod_2step()
        print('On line %i in function %s of module %s' % (main_lineno, main_fn, main_module))
        print('     Error on line %i in module %s' % (calling_lineno, calling_module))
        print('         Value "%s" not found in set: %s. Please use one of these values as input.' % (value, val_set))
        sys.exit('         ERROR: Incorrect parameter value.')

    return 0


def check_threshold(values=[], thresh=1.0, how='under'):
    """
    This function checks to see whether or not a numeric value is less than/equal to or greater than/equal to a
     given threshold value. If any are not, an error is returned, and the script is exited.

    Parameters:
    :param values: (list of any type) Values to check whether or not they are under or over a threshold, depending on
        the parameter 'how'
    :param thresh: (int/float) A numerical threshold to which to compare each value in values.
    :param how: (string) An option to test whether the values are 'under' (less than/equal to) or 'over' (greater
        than/equal to).
    :return: 0
    """
    # Check to make sure the arguments values and thresh are numeric (using check_numeric function in _err_check.py),
    # and that how is a string value, with 'over' or 'under' as the value.
    check_numeric(values=values)
    check_numeric(values=[thresh])
    check_string(values=[how])
    param_exists_in_set(value=how, val_set=['under', 'over'])

    for val in values:
        if how == 'under':
            if val <= thresh:
                pass
            else:
                main_module, main_fn, main_lineno = parent_fn_mod_3step()
                calling_module, calling_fn, calling_lineno = parent_fn_mod_2step()
                print('On line %i in function %s of module %s' % (main_lineno, main_fn, main_module))
                print('     Error at line %i in module %s' % (calling_lineno, calling_module))
                print('         Value (%0.2f) is over the maximum value of %0.2f' % (val, thresh))
                sys.exit('         ERROR: Parameter over maximum threshold.')
        elif how == 'over':
            if val >= thresh:
                pass
            else:
                main_module, main_fn, main_lineno = parent_fn_mod_3step()
                calling_module, calling_fn, calling_lineno = parent_fn_mod_2step()
                print('On line %i in function %s of module %s' % (main_lineno, main_fn, main_module))
                print('     Error at line %i in module %s' % (calling_lineno, calling_module))
                print('         Value (%0.2f) is under the minimum value of %0.2f' % (val, thresh))
                sys.exit('         ERROR: Parameter under minimum threshold.')

    return 0


# 2. Chemistry checking files
def is_element(eles, return_cleaned=False):
    """
    This function takes a list of strings that are presumed to be elements. It uses the package, "periodic" to check
        whether or not each member of the list is present in the table of elements.
    :param eles:
    :param return_cleaned: True/False value. If set to True, those elements that don't exist in the table of elements
        are removed from the list eles, and returned in eles_cln. Otherwise, nothing is retrned, and the script
        terminates if an unknown element is found.
    :return: 0 (int)/eles_cln (string list): eles_cln is the list [eles], but with those elements that do not exist
        in the periodic table elements removed. It is only returned if [return_cleaned]=True
    """
    eles_cln = []
    for ele, ele_num in zip(eles, range(0, len(eles))):
        if periodic.element(ele) is None:
            if return_cleaned:
                print('Element %s not found! Removed from elements list.' % ele)
                pass
            else:
                main_module, main_fn, main_lineno = parent_fn_mod_3step()
                calling_module, calling_fn, calling_lineno = parent_fn_mod_2step()
                print('On line %i in function %s of module %s' % (main_lineno, main_fn, main_module))
                print('     Error on line %i in module %s' % (calling_lineno, calling_module))
                print('         %s is not an element in the periodic table. Check input at index %i of list.'
                      % (ele, ele_num))
                sys.exit('         ERROR: Chemical element not found in periodic table.')
        elif (periodic.element(ele) is not None) and return_cleaned:
            eles_cln.append(ele)

    if return_cleaned:
        return eles_cln
    else:
        return 0


# 3. File existence/type checking
def f_is_csv(file=""):
    """
    This function checks 1) whether a file exists and 2) is a csv file or not, by checking the last 3 letters of the
        file name (i.e., if it ends with 'csv'). If it is not, returns False. If it is, returns True.
    :param file: (string) The file to be checked whether or not it is a csv file.
    :return: f_check: (boolean) True/False value indicating whether the file is a csv or not.
    """
    if os.path.isfile(path=file):
        ftype = file[-3:]
        if ftype.lower() == "csv":
            f_check = True
        else:
            f_check = False
    else:
        print('File not found')
        f_check = False

    return f_check


def f_is_xls(file=""):
    """
    This function checks 1) whether a file exists and 2) is an excel file (*.xls, *.xlsx) or not, by checking the
        last 3 letters of the file name (i.e., if it ends with 'csv'). If it is not, returns False. If it is,
        returns True.
    :param file: (string) The file to be checked whether or not it is a csv file.
    :return: f_check: (boolean) True/False value indicating whether the file is a csv or not.
    """
    if os.path.isfile(path=file):
        ftype1 = file[-3:]                          # Last 3 characters (for *.xls)
        ftype2 = file[-4:]                          # Last 4 characters (for *.xlsx)
        if ftype1.lower() == "xls":
            f_check = True
        elif ftype2.lower() == "xlsx":
            f_check = True
        else:
            f_check = False
    else:
        print('File not found')
        f_check = False

    return f_check


# 4. Error location finding functions
def parent_fn_mod_1step():
    """
    This function finds the calling module, function and line number 1 step prior to current function

    :return: calling module name, function name and line # for 1 step previous ('1step')
    """

    return stack()[1].filename, stack()[1].function, stack()[1].lineno

def parent_fn_mod_2step():
    """
    This function finds the calling module, function, and line number 2 steps prior to the current function.

    Parameters:
    :return: calling module name, function name, and line number for 2 previous function calls ('2step')
    """
    return stack()[2].filename, stack()[2].function, stack()[2].lineno


def parent_fn_mod_3step():
    """
    This function finds the calling module, function and line number 3 steps prior to the current function.

    Parameters:
    :return: calling module name, function name, and line number for 3 previous function calls ('3step')
    """
    return stack()[3].filename, stack()[3].function, stack()[3].lineno