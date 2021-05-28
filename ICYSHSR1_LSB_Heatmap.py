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
    filename = "./data/ARR0/NON_CORR_TEST_ALL-20210322-210304.hdf5"
    # ARR 1
    #filename = "./data/ARR1/NON_CORR_TEST_ALL-20210319-203909.hdf5"
    lsb = []
    res = []
    for i in range(49):
        tf = TransferFunctions(filename=filename, basePath="CHARTIER/ASIC0/MO/TDC/NON_CORR/ALL/FAST_255/SLOW_250/ARRAY_0", pixel_id=i*4, filter_lower_than=0.05)
        cur_lsb = 4000 / len(tf.ideal_tf)
        corr_resolution = np.std(tf.get_ideal() - tf.get_slope_corr_biased_linear())
        lsb.append(cur_lsb)
        res.append(corr_resolution)
        print(cur_lsb)
        print(corr_resolution)

        #print("TDC " + str(i) + " ratio " + str(ratio))
    print(lsb)
    print(res)
    with open('ratioLSB.pickle', 'wb') as f:
        pickle.dump([lsb, res], f)

main()
