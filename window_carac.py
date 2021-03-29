# coding=utf-8
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ICYSHSR1_transfer_function_ideal import TransferFunctions



def main():
    target_TDC = 2
    filename = "./data/ARR0/NON_CORR_TEST_ALL-20210322-210304.hdf5"
    tf = TransferFunctions(filename=filename, basePath="CHARTIER/ASIC0/MO/TDC/NON_CORR/ALL/FAST_255/SLOW_250/ARRAY_0", pixel_id=target_TDC*4, filter_lower_than=0)

    window_filename = "./data/ARR0/WINDOW_NON_CORR_Experiment-20210325-221604.hdf5"
    length_by_code = []
    for window_len in range(1, 11, 1):
        window_data_path = "CHARTIER/ASIC0/TDC/M0/1_TDC_ACTIVE/PLL/FAST_255/SLOW_250/NON_CORR/EXT/ADDR_" + str(target_TDC) + "/DELAY_0/WINDOW_LEN_" + str(window_len)
        with h5py.File(window_filename, "r") as h:
            ds = h[window_data_path]
            coarse = np.array(ds['Coarse'], dtype='int64')
            fine = np.array(ds['Fine'], dtype='int64')
            addr = np.array(ds['Addr'], dtype='int64')
            addr_filter = (addr == target_TDC*4)
            if not addr_filter.any():
                raise Exception("It seems like the targeted pixel was disabled or didn't trigger")
            coarses = coarse[addr_filter]
            fines = fine[addr_filter]
            timestamps = []
            for coarse, fine in zip(coarses, fines):
                timestamps.append(tf.code_to_timestamp(coarse, fine))

            diff = max(timestamps) - min(timestamps)
            length_by_code.append(diff)
            print("Win Len " + str(window_len) + ": Actual time: " + str(diff))

    plt.figure()
    plt.plot(range(len(length_by_code)), length_by_code, 'k--')
    plt.xlabel("Code définissant la largeur de la fenêtre")
    plt.ylabel("Durée de la fenêtre (ps)")
    plt.show()


main()
