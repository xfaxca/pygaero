# gen_chem.py

# Module/package import
import pandas as pd
import numpy as np
import pygaero._dchck as check
import sys
import periodic

__doc__ = "Module containing general chemistry functions to handle chemical formula names, \n" \
          "elemental analysis of formulas, etc."


# 1. Molecule name formatting/modification functions
def cln_molec_names(idx_names, delim='_'):
    """
    This function takes a list of strings (assumed to be molecule names for this package) and delimits them, using
        the user-specified delimiter. It was originally written for HR-ToF-CIMS data originating from Tofware in
         Igor Pro, so the default delimiter is an underscore ('_'). Returns the delimited list of strings.
    :param idx_names: (list of strings) A string list that should include molecule names (i.e., the time series
        column names).
    :param delim: (char) The character by which each string will be delimited.
    :return: cleaned_names: (list of strings) A numpy array containing the delimited names that were originally
        in parameter [idx_names]
    """
    # Check data types to make sure that there is no error in subsequent loops.
    if check.check_ls(ls=idx_names, nt_flag=True) and check.check_np_array(arr=idx_names, nt_flag=True):
        main_module, main_fn, main_lineno = check.parent_fn_mod_3step()
        calling_module, calling_fn, calling_lineno = check.parent_fn_mod_2step()
        print('On line %i in function %s of module %s' % (main_lineno, main_fn, main_module))
        print('     Error on line %i in module %s' % (calling_lineno, calling_module))
        print('         Invalid input for function %s' % calling_fn)
        sys.exit('ERROR: Either a list or np.array is required')
    check.check_string(values=idx_names)
    check.check_string(values=[delim])

    cleaned_names = []
    for i in range(0, len(idx_names)):
        # Clean name for output (if igor _ delimiters are in the name (comes as default most of the time)
        current_name = idx_names[i]
        # Check for first delim character and take everything to the right:
        if delim in current_name:
            current_name = idx_names[i].split(delim, 1)[1]
            # check for second delim character and take everything to the left
            if delim in current_name:
                current_name = current_name.split(delim, 1)[0]
        # append cleaned name to list
        cleaned_names.append(current_name)

        # print('cleaned names in function:', cleaned_namess)
    cleaned_names = np.array(cleaned_names)
    return cleaned_names


def replace_group(molec_ls, old_groups, new_group=""):
    """
    This function replaces a string sequence(s) within a molecule's name string with a new, user-specified group.
        An example of how this would be used is to remove cluster elements from an ion. For example, if the formula
        names for time series in a data set are of the form "CxHyOzI", then one can strip "I" from the equations
        to make the new names of the form "CxHyOz". This is done by setting [old_groups]="I" and [new_group]="". Another
        example would be to replace "NO4" and "NO5" in molecule names with "NOx" by setting [old_groups]=["NO4", "NO5"]
        and [new_group]="NOx"

    :param molec_ls: (list of strings) A string list of molecule names in which the character [old_group] is to be
        replaced by [new_group].
    :param old_groups: (list of strings) A list of strings for the old group which to to be replaced by the character(s)
        in [new_group]. Must be passed to the function as a list.
    :param new_group: (string) The character string for by which members of [old_groups] will be replaced.
    :return: molec_ls_new: (list of strings) A new string list with the characters switched out in each element
        according to the values of [old_ele] and [new_ele].
    """
    # Check data types to prevent errors in subsequent loops
    if check.check_ls(ls=molec_ls, nt_flag=True) and check.check_np_array(arr=molec_ls, nt_flag=True):
        main_module, main_fn, main_lineno = check.parent_fn_mod_3step()
        calling_module, calling_fn, calling_lineno = check.parent_fn_mod_2step()
        print('On line %i in function %s of module %s' % (main_lineno, main_fn, main_module))
        print('     Error on line %i in module %s' % (calling_lineno, calling_module))
        print('         Invalid input for function %s' % calling_fn)
        sys.exit('ERROR: Either a list or np.array is required')
    check.check_ls(ls=old_groups)
    for param in [old_groups, new_group, molec_ls]:
        check.check_string(values=param)

    molec_ls_new = []
    for item in molec_ls:
        new_item = item
        for group in old_groups:
            if group in item:
                new_item = new_item.replace(group, new_group)
            else:
                print('%s not found in %s' % (group, item))
        molec_ls_new.append(new_item)

    return molec_ls_new


# 2. Molecule elemental analysis and statistics functions
def ele_stats(molec_ls, ion_state=-1, cluster_group=None, clst_group_mw=126.90447, xtra_elements=None):
    """
    This function takes a list of string values that are chemical formulas and calculates a set of basic statistics for
        each formula. The statistics are then returned in a pandas DataFrame with the molecule names as the indices
        with each statistic as a column. Statistics include (column name listed):
        1. Basic elemental counts:
            C - # of C atoms in molecule
            H - # of H atoms in molecule
            O - # of O atoms in molecule
            N - # of N atoms in molecule
            Cl - # of Cl atoms in molecule
            F - # of F atoms in molecule
            Br - # of Br atoms in molecule
            [cluster_group] - # of user-defined cluster group in molecule (for cluster-forming ionization mechanisms)
            * More element counts can be added by using a list in the parameter [xtra_elements]
        2. O/C and H/C:
            O/C - oxygen to carbon ratio
            H/C - hydrogen to carbon ratio
        3. Oxidation State (Kroll et al., 2011):
            OSC: 2*(O/C) - (H/C)
            OSC_N: 2*(O/C) - (H/C) - 5*(N/C). Assumes all nitrogen are nitrate groups (see reference for details)
        4. Molecular weights:
            MW: Molecular weight for exact formula in molecule name string (in molec_ls)
            MW_xclust: Molecular weight without the cluster group specified by cluster_group
    :param molec_ls: (list of strings) A list of strings which are molecule names for which statistics will be
        calculated.
    :param ion_state: (int or float) The multiplier of the mass of electrons (mol/g) to add/subtract for a charged
        molecule's molecular mass (in mass_electron*(-1)*ion_state). ion_state is equal to the charge on each ion
        (ion_state=0 for neutral)
    :param cluster_group: (string) If ions are in a clustered form (e.g., clustered with I-), the user should specify
        what element (or group) is the cluster group so that it can be counted. Note that if cluster_group is an
        element, it should NOT be repeated in the list, xtra_elements.
    :param clst_group_mw: (int or float) Molecular weight/mass of the cluster group in g/mol. Important to specify this for an
        accurate molecular weight calculations. If it is an element, then clst_group_mw can be set to 0 and the
        correct elemental molecular weight will be used in MW calculations.
    :param xtra_elements: (list of strings) Extra elements which should be accunted for. That is, elements other
        than C, H, O, N, Cl, F, and Br.
    :return df_ele: A returned pandas DataFrame containing statistics for the molecules in molec_ls
    """

    # Check data types to prevent subsequent errors.
    if check.check_ls(ls=molec_ls, nt_flag=True) and check.check_np_array(arr=molec_ls, nt_flag=True):
        main_module, main_fn, main_lineno = check.parent_fn_mod_3step()
        calling_module, calling_fn, calling_lineno = check.parent_fn_mod_2step()
        print('On line %i in function %s of module %s' % (main_lineno, main_fn, main_module))
        print('     Error on line %i in module %s' % (calling_lineno, calling_module))
        print('         Invalid input for function %s' % calling_fn)
        sys.exit('ERROR: Either a list or np.array is required')
    if (check.check_ls(ls=xtra_elements, nt_flag=True) and check.check_np_array(arr=xtra_elements, nt_flag=True)) \
            and xtra_elements is not None:
        main_module, main_fn, main_lineno = check.parent_fn_mod_3step()
        calling_module, calling_fn, calling_lineno = check.parent_fn_mod_2step()
        print('On line %i in function %s of module %s' % (main_lineno, main_fn, main_module))
        print('     Error on line %i in module %s' % (calling_lineno, calling_module))
        print('         Invalid input for function %s' % calling_fn)
        sys.exit('ERROR: Either a list or np.array is required')
    # print('check.check_ls(ls=cluster_group), nt_flag=True', check.check_ls(ls=cluster_group, nt_flag=True))
    if not cluster_group:
        pass
    elif not check.check_ls(ls=cluster_group, nt_flag=True):
        main_module, main_fn, main_lineno = check.parent_fn_mod_3step()
        calling_module, calling_fn, calling_lineno = check.parent_fn_mod_2step()
        print('On line %i in function %s of %s' % (main_lineno, main_fn, main_module))
        print('     Error on line %i in module %s' % (calling_lineno, calling_module))
        print('         Invalid input for function %s' % calling_fn)
        sys.exit("Inappropriate type passed to parameter: list.")
    # elif cluster_group is not None:
    else:
        check.check_string(values=[cluster_group])
    check.check_numeric(values=[ion_state])
    check.check_numeric(values=[clst_group_mw])

    # Set column names for returned DataFrame, df_ele.
    # columns = ["C", "H", "O", "N", "Cl", "F", "Br", cluster_group, "O/C", "H/C", "OSC", "OSC_N",
    #            "MW", "MW_xclust"]
    columns = ["C", "H", "O", "N", "Cl", "F", "Br", "O/C", "H/C", "OSC", "OSC_N",
               "MW", "MW_xclust"]
    if cluster_group is not None:
        print('extending by cluster_group')
        columns.extend(cluster_group)
    # Extend columns list by # of elements in xtra_elements.
    if xtra_elements is not None:
        xtra_elements = check.is_element(eles=xtra_elements, return_cleaned=True)
        columns.extend(xtra_elements)
    columns = remove_duplicates(values=columns)

    # Define me, the molar mass of electrons to adjust molar mass of ion molecules by their respective charge
    me = 0.00054857990946
    df_ele = pd.DataFrame(index=molec_ls,
                          columns=columns,
                          data=0.0)
    base_elements = ["C", "H", "O", "N", "Cl", "F", "Br"]
    if xtra_elements is not None:
        base_elements.extend(xtra_elements)
    elements_ls = base_elements

    for molec in molec_ls:
        for char_num in range(0, len(molec)):
            if molec[char_num].isdigit():
                pass
            elif molec[char_num].isalpha():
                for ele in elements_ls:
                    if molec[char_num] == ele:
                        if len(molec) > (char_num + 1):
                            if molec[char_num+1].isdigit():
                                ele_count = int(molec[char_num+1])
                                if len(molec) > (char_num + 2):
                                    if molec[char_num+2].isdigit():
                                        ele_count = ele_count*10 + int(molec[char_num+2])
                            else:
                                ele_count = 1
                        else:
                            ele_count = 1
                        df_ele.ix[molec, ele] = ele_count

        # Loop to count cluster groups in the element
        if (cluster_group is not None) and (cluster_group in molec):
            clst_idx = molec.index(cluster_group)
            next_char_idx = clst_idx + len(cluster_group)
            if len(molec) > next_char_idx:
                if molec[next_char_idx].isdigit():
                    clst_count = int(molec[next_char_idx])
                    if len(molec) > (next_char_idx + 1):
                        clst_count = clst_count * 10 + int(molec[next_char_idx + 1])
                else:
                    clst_count = 1
            else:
                clst_count = 1
        else:
            clst_count = 0
        df_ele.ix[molec, cluster_group] = clst_count

        # Calculate molecular weight/mass now that all basic elements/cluster groups have been counted
        MW = 0.0
        MW_xclust = 0.0
        for ele in elements_ls:
            ele_stats = periodic.element(ele)
            MW += ele_stats.mass * df_ele.ix[molec, ele]
            MW_xclust += ele_stats.mass * df_ele.ix[molec, ele]
        if (cluster_group is not None) and (len(cluster_group) > 0):
            if clst_group_mw == 0:
                clst_stats = periodic.element(cluster_group)
                if clst_stats is None:
                    print('Cluster group "%s" not found in periodic table. If it is not an element, please define'
                          'a molecular weight (g/mol) for the group using parameter [clst_group_mw] in '
                          'function ele_stats()' % cluster_group)
                else:
                    MW += clst_stats.mass * clst_count
            else:
                MW += clst_group_mw * clst_count
        # Adjust MW by weight of electron (with respect to parameter [ion_state])
        MW += ion_state*(-1)*me
        df_ele.ix[molec, "MW"] = MW
        df_ele.ix[molec, "MW_xclust"] = MW_xclust

    # Calc all O/C, H/C and Oxidation states and then assign to df_ele
    df_ele.ix[:, "O/C"] = o_to_c(o=df_ele.ix[:, "O"].values, c=df_ele.ix[:, "C"].values)
    df_ele.ix[:, "H/C"] = h_to_c(h=df_ele.ix[:, "H"].values, c=df_ele.ix[:, "C"].values)
    df_ele.ix[:, "OSC"] = osc(c=df_ele.ix[:, "C"].values, h=df_ele.ix[:, "H"].values,
                              o=df_ele.ix[:, "O"].values)
    df_ele.ix[:, "OSC_N"] = osc_nitr(c=df_ele.ix[:, "C"].values, h=df_ele.ix[:, "H"].values,
                                     o=df_ele.ix[:, "O"].values, n=df_ele.ix[:, "N"].values)

    return df_ele


def o_to_c(o=[], c=[]):
    """
    This function calculates a simple O/C ratio for a list of oxygen and carbon numbers. len(o) and len(c) must
        be equal.
    :param o: (int/float list) A list of numeric values representing the number of oxygens for a list of molecules. Can be float
        or int, but float should only be used if the calculation is for a bulk average. Otherwise, a float value
        doesn't make sense for a single molecule.
    :param c: (int/float list) A list of numeric values representing the number of oxygens for a list of molecules. Can be float
        or int, but float should only be used if the calculation is for a bulk average. Otherwise, a float value
        doesn't make sense for a single molecule.
    :return: oc_ratios: (float list) A list of float values that are the index-to-index ratios of the values in o and c
    """
    # Check to make sure that input lists are numeric and the same length to prevent errors during processing.
    if (check.check_ls(ls=o, nt_flag=True) and check.check_np_array(arr=o, nt_flag=True)) or \
            (check.check_ls(ls=c, nt_flag=True) and check.check_np_array(arr=c)):
        main_module, main_fn, main_lineno = check.parent_fn_mod_3step()
        calling_module, calling_fn, calling_lineno = check.parent_fn_mod_2step()
        print('On line %i in function %s of module %s' % (main_lineno, main_fn, main_module))
        print('     Error on line %i in module %s' % (calling_lineno, calling_module))
        print('         Invalid input for function %s' % calling_fn)
        sys.exit('ERROR: Either a list or np.array is required')
    check.check_eq_ls_len(list_ls=[o, c])
    check.check_numeric(values=o)
    check.check_numeric(values=c)

    oc_ratios = []
    for no, nc in zip(o, c):
        if nc == 0:
            oc_ratios.append(np.nan)
        else:
            oc_ratios.append(float(no / nc))

    return oc_ratios


def h_to_c(h=[], c=[]):
    """
    This function calculates a simple O/C ratio for a list of oxygen and carbon numbers. len(h) and len(c) must
        be equal.
    :param h: (int/float list) A list of numeric values representing the number of hydrogens for a list of molecules.
        Can be float or int, but float should only be used if the calculation is for a bulk average. Otherwise, a float
        value doesn't make sense for a single molecule.
    :param c: (int/float list) A list of numeric values representing the number of hydrogens for a list of molecules.
        Can be float or int, but float should only be used if the calculation is for a bulk average. Otherwise, a float
        value doesn't make sense for a single molecule.
    :return: hc_ratios: (float list) A list of values that are the index-to-index ratios of the values in h and c
    """
    # Check to make sure that input lists are numeric and the same length to prevent errors during processing.
    if (check.check_ls(ls=h, nt_flag=True) and check.check_np_array(arr=h, nt_flag=True)) or \
            (check.check_ls(ls=c, nt_flag=True) and check.check_np_array(arr=c)):
        main_module, main_fn, main_lineno = check.parent_fn_mod_3step()
        calling_module, calling_fn, calling_lineno = check.parent_fn_mod_2step()
        print('On line %i in function %s of module %s' % (main_lineno, main_fn, main_module))
        print('     Error on line %i in module %s' % (calling_lineno, calling_module))
        print('         Invalid input for function %s' % calling_fn)
        sys.exit('ERROR: Either a list or np.array is required')
    check.check_eq_ls_len(list_ls=[h, c])
    check.check_numeric(values=h)
    check.check_numeric(values=c)

    hc_ratios = []
    for no, nc in zip(h, c):
        if nc == 0:
            hc_ratios.append(np.nan)
        else:
            hc_ratios.append(float(no / nc))

    return hc_ratios


def osc(c=[], h=[], o=[]):
    """
    This function calculates a carbon oxidation state (OSC from Kroll et al., 2011). This is calculated by the formula:
        OSC = 2*(O/C) - (H/C).
    :param c: (int/float list) Numerical list with the number of carbons in the molecules for which OSC will be
        calculated.
    :param h: (int/float list) Numerical list with the number of hydrogens in the molecules for which OSC will be
        calculated.
    :param o: (int/float list) Numerical list with the number of oxygens in the molecules for which OSC will be
        calculated.
    :return: ox_states: (float list) A list of values that are the carbon oxidation state calculated from the index-by-
        index values of C, H and O
    """
    # Verify that parameters are lists of numerical values (numpy ndarray or python list) to prevent error in
    # subsequent loops
    if (check.check_ls(ls=c, nt_flag=True) and check.check_np_array(arr=c, nt_flag=True)) or \
            (check.check_ls(ls=h, nt_flag=True) and check.check_np_array(arr=h)) or \
            (check.check_ls(ls=o, nt_flag=True) and check.check_np_array(arr=o)):
        main_module, main_fn, main_lineno = check.parent_fn_mod_3step()
        calling_module, calling_fn, calling_lineno = check.parent_fn_mod_2step()
        print('On line %i in function %s of module %s' % (main_lineno, main_fn, main_module))
        print('     Error on line %i in module %s' % (calling_lineno, calling_module))
        print('         Invalid input for function %s' % calling_fn)
        sys.exit('ERROR: Either a list or np.array is required')
    check.check_eq_ls_len(list_ls=[c, h, o])
    check.check_numeric(values=c)
    check.check_numeric(values=h)
    check.check_numeric(values=o)

    ox_states = []
    for nc, nh, no in zip(c, h, o):
        if nc == 0:
            ox_states.append(np.nan)
        else:
            ox_states.append(float((2*(no / nc) - (nh / nc))))

    return ox_states


def osc_nitr(c=[], h=[], o=[], n=[]):
    """
    This function calculates a carbon oxidation state (OSC from Kroll et al., 2011). This is calculated by the formula:
        OSC = 2*(O/C) - (H/C) - 5*(N/C).
    :param c: (int/float list) Numerical list with the number of carbons in the molecules for which OSC will be
        calculated.
    :param h: (int/float list) Numerical list with the number of hydrogens in the molecules for which OSC will be
        calculated.
    :param o: (int/float list) Numerical list with the number of oxygens in the molecules for which OSC will be
        calculated.
    :param n: (int/float list) Numerical list with the number of nitrogens in the molecules for which OSC will be
        calculated.
    :return: ox_states_nitr: (float list) Oxidation states, accounting for nitrogen groups (assumed nitrates)
    """
    # Verify that parameters are lists of numerical values (numpy ndarray or python list) to prevent error in
    # subsequent loops
    if (check.check_ls(ls=c, nt_flag=True) and check.check_np_array(arr=c, nt_flag=True)) or \
            (check.check_ls(ls=h, nt_flag=True) and check.check_np_array(arr=h)) or \
            (check.check_ls(ls=o, nt_flag=True) and check.check_np_array(arr=o)) or \
            (check.check_ls(ls=n, nt_flag=True) and check.check_np_array(arr=n)):
        main_module, main_fn, main_lineno = check.parent_fn_mod_3step()
        calling_module, calling_fn, calling_lineno = check.parent_fn_mod_2step()
        print('On line %i in function %s of module %s' % (main_lineno, main_fn, main_module))
        print('     Error on line %i in module %s' % (calling_lineno, calling_module))
        print('         Invalid input for function %s' % calling_fn)
        sys.exit('ERROR: Either a list or np.array is required')
    check.check_eq_ls_len(list_ls=[c, h, o, n])
    check.check_numeric(values=c)
    check.check_numeric(values=h)
    check.check_numeric(values=o)
    check.check_numeric(values=n)

    ox_states_nitr = []
    for nc, nh, no, nn in zip(c, h, o, n):
        if nc == 0:
            ox_states_nitr.append(np.nan)
        else:
            ox_states_nitr.append(float((2*(no / nc) - (nh / nc) - 5*(nn / nc))))

    return ox_states_nitr


# Miscellaneous functions
def remove_duplicates(values):
    """
    Function to remove duplcate values from a list
    :param values: (int/float/str/char) values from which duplicates should be removed
    :return:
    """
    values_rm = []
    seen = set()
    for val in values:
        if val not in seen:
            values_rm.append(val)
            seen.add(val)
    return values_rm
