# coding=utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
from ICYSHSR1_transfer_function_ideal import TransferFunctions
from transfer_function_no_correction import TransferFunctionNoCorrections
from transfer_function_ICSSHSR4 import TransferFunctionICSSHSR4
from transfer_function_ICYSHSR1 import TransferFunctionICYSHSR1

# TDC_VISUALIZED = 2
TOTAL_PERIOD = 4000




def main():
    # ARR 0
    # Fichier 40 go
    #filename = "./data/ARR0/NON_CORR_TEST_ALL-20210322-210304.hdf5"
    #path = "CHARTIER/ASIC0/MO/TDC/NON_CORR/ALL/FAST_255/SLOW_250/ARRAY_0"

    # Fichier Pascal
    filename = "./data/ARR0/EXP_NON_CORR_CALIB_COEF.hdf5"
    path = "CHARTIER/ASIC0/TDC/M0/ALL_TDC_ACTIVE/PLL/FAST_255/SLOW_250/NON_CORR/EXT/ADDR_ALL/RAW"

    # Fichier Mich
    #filename = "./data/ARR0/NON_CORR_TEST_ALL-20210414-212030.hdf5"
    #path = "CHARTIER/ASIC0/TDC/M0/ALL_TDC_ACTIVE/PLL/FAST_255/SLOW_250/NON_CORR/EXT/ADDR_ALL/RAW"
    # ARR 1
    #filename = "./data/ARR1/NON_CORR_TEST_ALL-20210319-203909.hdf5"
    coefficients_lin = {}
    coefficients_lin_bias = {}
    coefficients_lin_bias_slope = {}
    for i in range(49):
        tf = TransferFunctions(filename=filename, basePath=path, pixel_id=i*4, filter_lower_than=0.05)

        tf.linear_regression_algorithm()
        coefficients_lin[i] = tf.get_coefficients()

        tf.linear_regression_algorithm(True, False)
        coefficients_lin_bias[i] = tf.get_coefficients()

        tf.linear_regression_algorithm(True, True)
        coefficients_lin_bias_slope[i] = tf.get_coefficients()

    print(coefficients_lin)
    with open('20may_corr_coef_lin.pickle', 'wb') as f:
        pickle.dump(coefficients_lin, f)

    print(coefficients_lin_bias)
    with open('20may_corr_coef_lin_bias.pickle', 'wb') as f:
        pickle.dump(coefficients_lin_bias, f)

    print(coefficients_lin_bias_slope)
    with open('20may_corr_coef_lin_bias_slope.pickle', 'wb') as f:
        pickle.dump(coefficients_lin_bias_slope, f)

main()
