import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from transfer_function_ideal import TransferFunctionIdeal


def main():
    filename = "./../data/Demo_Uncorrelated_coefficients_4.txt"
    tf_ideal = TransferFunctionIdeal(filename=filename)


    tf_graph = tf_ideal.get_transfer_functions_raw_data()
    inl = tf_ideal.get_inl_data()
    dnl = tf_ideal.get_dnl_data()
    fig, ax = plt.subplots()
    #ax.plot(tf_graph[0][1], tf_graph[1][1])
    #ax.plot(inl[0][1], inl[1][1])
    ax.plot(dnl[0][1], dnl[1][1])

    plt.show()

main()
