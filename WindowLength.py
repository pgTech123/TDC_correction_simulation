# coding=utf-8
import math
import h5py
import numpy as np


def main():
    filename = "./data/Window/GOOD_ALL_127-128-Window_Size_Experiment-20210609-150059.hdf5"
    path1 = "CHARTIER/ASIC0/TDC/M0/ALL_TDC_ACTIVE/PLL/FAST_"
    path2 = "/SLOW_"
    path3 = "/WINDOW_LENGTH/EXT/ADDR_ALL/RAW"

    window_length = 128

    with h5py.File(filename, "r") as h:
        path = path1 + str(window_length)
        delays = h[path].keys()
        for delay_path in delays:
            path = path1 + str(window_length) + "/" + delay_path + path3
            packet = np.array(h[path])
            data = packet[packet != 0xAAAAAAAAAAAAAAAA]
            data = data[data != 0xAAAAAAABAAAAAAAB]
            data = data[data != 0]
            data = data[data != 327680]
            data = data[data != 327681]
            if len(data) != 0:
                print(delay_path)


main()