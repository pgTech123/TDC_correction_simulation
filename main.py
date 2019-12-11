import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from transfer_function_ideal import TransferFunctionIdeal
from transfer_function_no_correction import TransferFunctionNoCorrections
from transfer_function_ICSSHSR4 import TransferFunctionICSSHSR4
from transfer_function_ICYSHSR1 import TransferFunctionICYSHSR1

TDC_VISUALIZED = 3

def main():
    filename = "./../data/Demo_Uncorrelated_coefficients_4.txt"
    tf_ideal = TransferFunctionIdeal(filename=filename)
    tf_icsshsr4_median = TransferFunctionICSSHSR4(tf_ideal, algorithm="median")
    tf_icsshsr4_linear_reg = TransferFunctionICSSHSR4(tf_ideal, algorithm="linear_regression")
    tf_icyshsr1 = TransferFunctionICYSHSR1(tf_ideal)

    tf_graph_ideal = tf_ideal.get_transfer_functions_raw_data()
    tf_graph_icsshsr4_median = tf_icsshsr4_median.get_transfer_functions_raw_data()
    tf_graph_icsshsr4_linear_reg = tf_icsshsr4_linear_reg.get_transfer_functions_raw_data()
    tf_graph_icyshsr1 = tf_icyshsr1.get_transfer_functions_raw_data()

    plt.figure()
    plt.plot(tf_graph_ideal[0][1], tf_graph_ideal[1][1])
    plt.plot(tf_graph_icsshsr4_median[0][1], tf_graph_icsshsr4_median[1][1])
    plt.plot(tf_graph_icsshsr4_linear_reg[0][1], tf_graph_icsshsr4_linear_reg[1][1])
    plt.plot(tf_graph_icyshsr1[0][1], tf_graph_icyshsr1[1][1])

    plt.figure()
    plt.plot(tf_graph_ideal[0][TDC_VISUALIZED], tf_graph_ideal[1][TDC_VISUALIZED]-tf_graph_icsshsr4_linear_reg[1][TDC_VISUALIZED])
    plt.plot(tf_graph_ideal[0][TDC_VISUALIZED], tf_graph_ideal[1][TDC_VISUALIZED]-tf_graph_icsshsr4_median[1][TDC_VISUALIZED])
    plt.plot(tf_graph_ideal[0][TDC_VISUALIZED], tf_graph_ideal[1][TDC_VISUALIZED]-tf_graph_icyshsr1[1][TDC_VISUALIZED])

    min_ideal_icsshsr4_median = min(tf_graph_ideal[1][TDC_VISUALIZED]-tf_graph_icsshsr4_median[1][TDC_VISUALIZED])
    max_ideal_icsshsr4_median = max(tf_graph_ideal[1][TDC_VISUALIZED]-tf_graph_icsshsr4_median[1][TDC_VISUALIZED])

    min_ideal_icsshsr4_linear_reg = min(tf_graph_ideal[1][TDC_VISUALIZED]-tf_graph_icsshsr4_linear_reg[1][TDC_VISUALIZED])
    max_ideal_icsshsr4_linear_reg = max(tf_graph_ideal[1][TDC_VISUALIZED]-tf_graph_icsshsr4_linear_reg[1][TDC_VISUALIZED])

    min_ideal_icyshsr1 = min(tf_graph_ideal[1][TDC_VISUALIZED]-tf_graph_icyshsr1[1][TDC_VISUALIZED])
    max_ideal_icyshsr1 = max(tf_graph_ideal[1][TDC_VISUALIZED]-tf_graph_icyshsr1[1][TDC_VISUALIZED])

    print("Error range on correction for ICSSHSR4 median: " + str(max_ideal_icsshsr4_median - min_ideal_icsshsr4_median))
    print("Error range on correction for ICSSHSR4 linear regression: " + str(max_ideal_icsshsr4_linear_reg - min_ideal_icsshsr4_linear_reg))
    print("Error range on correction for ICYSHSR1: " + str(max_ideal_icyshsr1 - min_ideal_icyshsr1))

    inl = tf_ideal.get_inl_data()
    dnl = tf_ideal.get_dnl_data()

    #plt.figure()
    #plt.plot(inl[0][TDC_VISUALIZED], inl[1][TDC_VISUALIZED])
    #plt.plot(dnl[0][TDC_VISUALIZED], dnl[1][TDC_VISUALIZED])

    plt.show()


main()
