import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from transfer_function_ideal import TransferFunctionIdeal
from transfer_function_ICSSHSR4 import TransferFunctionICSSHSR4


def main():
    fig, ax = plt.subplots()

    filename = "./../data/Demo_Uncorrelated_coefficients_4.txt"
    tf_ideal = TransferFunctionIdeal(filename=filename)
    # Note: The linear regression algorithm seems to give better results
    tf_icsshsr4 = TransferFunctionICSSHSR4(tf_ideal, algorithm="linear_regression")

    tf_graph_ideal = tf_ideal.get_transfer_functions_raw_data()
    tf_graph_icsshsr4 = tf_icsshsr4.get_transfer_functions_raw_data()

    #ax.plot(tf_graph_ideal[0][1], tf_graph_ideal[1][1])
    #ax.plot(tf_graph_icsshsr4[0][1], tf_graph_icsshsr4[1][1])

    ax.plot(tf_graph_ideal[0][1], tf_graph_ideal[1][1]-tf_graph_icsshsr4[1][1])

    inl = tf_ideal.get_inl_data()
    dnl = tf_ideal.get_dnl_data()
    #ax.plot(tf_graph[0][1], tf_graph[1][1])
    #ax.plot(inl[0][1], inl[1][1])
    #ax.plot(dnl[0][1], dnl[1][1])

    plt.show()

main()
