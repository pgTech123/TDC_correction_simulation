# coding=utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ICYSHSR1_transfer_function_ideal import TransferFunctions
from transfer_function_no_correction import TransferFunctionNoCorrections
from transfer_function_ICSSHSR4 import TransferFunctionICSSHSR4
from transfer_function_ICYSHSR1 import TransferFunctionICYSHSR1

# TDC_VISUALIZED = 2
TOTAL_PERIOD = 4000


def print_stats(target_graph, ideal_graph, name):
    diff = ideal_graph-target_graph
    min_diff = min(diff)
    max_diff = max(diff)

    print("Error range on correction for " + str(name) + ": " + str(max_diff - min_diff) +
          ", max=" + str(max_diff) + ", min=" + str(min_diff))
    print("Stddev for " + str(name) + ": " + str(np.std(diff)))



def main():
    # ARR 0
    filename = "/media/pascal/Files/Maitrise/ICYSHSR1/Data/ARR0/NON_CORR_TEST_ALL-20210322-210304.hdf5"
    # ARR 1
    #filename = "./data/ARR1/NON_CORR_TEST_ALL-20210319-203909.hdf5"

    tf = TransferFunctions(filename=filename, basePath="CHARTIER/ASIC0/MO/TDC/NON_CORR/ALL/FAST_255/SLOW_250/ARRAY_0", pixel_id=12)
    #tf = TransferFunctions(filename=filename, basePath="CHARTIER/ASIC0/TDC/NON_CORR/FAST_255/SLOW_250/ARRAY_0/ADDR_13", pixel_id=52)

    plt.figure()
    plt.plot(tf.density_code)
    plt.xlabel("Code du CTN")
    plt.ylabel("Nombre de compte")

    plt.figure()
    plt.plot(tf.get_ideal(), 'k--',label="Fonction de transfert idéale")
    #plt.plot(tf.get_median(), 'g', label="Pente médiane")
    #plt.plot(tf.get_linear(), 'r', label="Régression linéaire")
    #plt.plot(tf.get_biased_linear(), 'b', label="ICSSHSRY Algorithm: Bias correction on each coarse")
    #plt.plot(tf.get_slope_corr_biased_linear(), 'm', label="ICSSHSRY Algorithm: Bias and slope correction on each coarse")
    plt.xlabel("Code du CTN")
    plt.ylabel("Temps depuis le dernier coup d'horloge (ps)")
    plt.legend()

    plt.figure()
    plt.plot(range(len(tf.get_ideal())), np.zeros(len(tf.get_ideal())), 'k--', label="Idéal")
    plt.plot(range(len(tf.get_ideal())), tf.get_ideal()-tf.get_linear(), 'r', label="Régression linéaire")
    plt.plot(range(len(tf.get_ideal())), tf.get_ideal()-tf.get_median(), 'g', label="Pente médiane")
    plt.plot(range(len(tf.get_ideal())), tf.get_ideal()-tf.get_biased_linear(), 'b', label="Correction du décallage pour chaque code grossier")
    plt.plot(range(len(tf.get_ideal())), tf.get_ideal()-tf.get_slope_corr_biased_linear(), 'm', label="Correction du décallage et de la pente pour chaque code grossier")
    # plt.title("Error between the ideal transfer function and different correction algorithms")
    #plt.title("Erreur entre la fonction de transfert idéale et différents algorithmes de correction")
    plt.xlabel("Code du CTN")
    plt.ylabel("Temps depuis le dernier coup d'horloge (ps)")
    plt.legend()

    print_stats(tf.get_median(), tf.get_ideal(), "ICSSHSR4 median")
    print_stats(tf.get_linear(), tf.get_ideal(), "ICSSHSR4 linear regression")
    print_stats(tf.get_biased_linear(), tf.get_ideal(), "ICYSHSR1")
    print_stats(tf.get_slope_corr_biased_linear(), tf.get_ideal(), "ICYSHSR1 better")

    plt.show()




"""
    tf_icsshsr4_median = TransferFunctionICSSHSR4(tf_ideal, algorithm="median")
    tf_icsshsr4_linear_reg = TransferFunctionICSSHSR4(tf_ideal, algorithm="linear_regression")
    tf_icyshsr1 = TransferFunctionICYSHSR1(tf_ideal, "lookup_coarse")
    tf_icyshsr1_better = TransferFunctionICYSHSR1(tf_ideal, "lookup_and_fine_correction")


    tf_graph_ideal = tf_ideal.get_transfer_functions_raw_data()
    tf_graph_icsshsr4_median = tf_icsshsr4_median.get_transfer_functions_raw_data()
    tf_graph_icsshsr4_linear_reg = tf_icsshsr4_linear_reg.get_transfer_functions_raw_data()
    tf_graph_icyshsr1 = tf_icyshsr1.get_transfer_functions_raw_data()
    tf_graph_icyshsr1_better = tf_icyshsr1_better.get_transfer_functions_raw_data()

    # Transfer function graph
    plt.figure()
    #plt.plot(tf_graph_ideal[0][TDC_VISUALIZED], tf_graph_ideal[1][TDC_VISUALIZED], 'k--', label="Ideal Transfer Function")
    plt.plot(tf_graph_ideal[0][TDC_VISUALIZED], tf_graph_ideal[1][TDC_VISUALIZED], 'k--', label="Fonction de transfert idéale")
    #plt.plot(tf_graph_icsshsr4_median[0][TDC_VISUALIZED], tf_graph_icsshsr4_median[1][TDC_VISUALIZED], 'g', label="ICSSHSR4 Algorithm: Median Slope")
    plt.plot(tf_graph_icsshsr4_median[0][TDC_VISUALIZED], tf_graph_icsshsr4_median[1][TDC_VISUALIZED], 'g', label="Pente médiane")
    #plt.plot(tf_graph_icsshsr4_linear_reg[0][TDC_VISUALIZED], tf_graph_icsshsr4_linear_reg[1][TDC_VISUALIZED], 'r', label="ICSSHSR4 Algorithm: Linear Regression")
    plt.plot(tf_graph_icsshsr4_linear_reg[0][TDC_VISUALIZED], tf_graph_icsshsr4_linear_reg[1][TDC_VISUALIZED], 'r', label="Régression linéaire")
    #plt.plot(tf_graph_icyshsr1[0][TDC_VISUALIZED], tf_graph_icyshsr1[1][TDC_VISUALIZED], 'b', label="ICSSHSRY Algorithm: Bias correction on each coarse")
    #plt.plot(tf_graph_icyshsr1_better[0][TDC_VISUALIZED], tf_graph_icyshsr1_better[1][TDC_VISUALIZED], 'm', label="ICSSHSRY Algorithm: Bias and slope correction on each coarse")
    # plt.title("Transfer function of the time conversion for different algorithms")
    #plt.title("Fonction de transfert de la conversion en temps selon différents coefficients")
    plt.xlabel("Code du CTN")
    plt.ylabel("Temps depuis le dernier coup d'horloge (ps)")
    plt.legend()

    # Error graph
    plt.figure()
    plt.plot(tf_graph_ideal[0][TDC_VISUALIZED], np.zeros(len(tf_graph_ideal[0][TDC_VISUALIZED])), 'k--', label="Idéal")
    plt.plot(tf_graph_ideal[0][TDC_VISUALIZED], tf_graph_ideal[1][TDC_VISUALIZED]-tf_graph_icsshsr4_linear_reg[1][TDC_VISUALIZED], 'r', label="Régression linéaire")
    plt.plot(tf_graph_ideal[0][TDC_VISUALIZED], tf_graph_ideal[1][TDC_VISUALIZED]-tf_graph_icsshsr4_median[1][TDC_VISUALIZED], 'g', label="Pente médiane")
    plt.plot(tf_graph_ideal[0][TDC_VISUALIZED], tf_graph_ideal[1][TDC_VISUALIZED]-tf_graph_icyshsr1[1][TDC_VISUALIZED], 'b', label="Correction du décallage pour chaque code grossier")
    plt.plot(tf_graph_ideal[0][TDC_VISUALIZED], tf_graph_ideal[1][TDC_VISUALIZED]-tf_graph_icyshsr1_better[1][TDC_VISUALIZED], 'm', label="Correction du décallage et de la pente pour chaque code grossier")
    # plt.title("Error between the ideal transfer function and different correction algorithms")
    #plt.title("Erreur entre la fonction de transfert idéale et différents algorithmes de correction")
    plt.xlabel("Code du CTN")
    plt.ylabel("Temps depuis le dernier coup d'horloge (ps)")
    plt.legend()

    print_stats(tf_graph_icsshsr4_median[1][TDC_VISUALIZED], tf_graph_ideal[1][TDC_VISUALIZED], "ICSSHSR4 median")
    print_stats(tf_graph_icsshsr4_linear_reg[1][TDC_VISUALIZED], tf_graph_ideal[1][TDC_VISUALIZED], "ICSSHSR4 linear regression")
    print_stats(tf_graph_icyshsr1[1][TDC_VISUALIZED], tf_graph_ideal[1][TDC_VISUALIZED], "ICYSHSR1")
    print_stats(tf_graph_icyshsr1_better[1][TDC_VISUALIZED], tf_graph_ideal[1][TDC_VISUALIZED], "ICYSHSR1 better")

    plt.figure()
    histogram = tf_ideal.get_histograms()[TDC_VISUALIZED]
    plt.bar(x=histogram[1][:-1], height=histogram[0])

    inl = tf_ideal.get_inl_data()
    dnl = tf_ideal.get_dnl_data()

    #plt.figure()
    #plt.plot(inl[0][TDC_VISUALIZED], inl[1][TDC_VISUALIZED])
    #plt.plot(dnl[0][TDC_VISUALIZED], dnl[1][TDC_VISUALIZED])

    plt.show()

"""
main()
