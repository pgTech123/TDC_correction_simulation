# coding=utf-8
import math
import h5py
import numpy as np
import matplotlib.pyplot as plt



def main():
    fig, ax = plt.subplots()
    filename = "./data/Window/Window_Size_Experiment-20210522-150932.hdf5"
    path1 = "CHARTIER/ASIC0/TDC/M0/ALL_TDC_ACTIVE/PLL/FAST_"
    path2 = "/SLOW_"
    path3 = "/WINDOW_LENGTH/EXT/ADDR_ALL/RAW"

    window_length = 200
    pixels = np.zeros(196)
    with h5py.File(filename, "r") as h:
        path = path1 + str(window_length)
        delays = h[path].keys()
        for delay_path in delays:
            index = 0
            path = path1 + str(window_length) + "/" + delay_path + path3
            data = np.array(h[path])
            for packet in data:
                if packet == 0xAAAAAAAAAAAAAAAA:
                    index = 0
                elif packet == 0xAAAAAAABAAAAAAAB:
                    index = 0
                elif packet == 327680:
                    pass
                else:
                    for offset in range(8):
                        count = (int(packet) >> (8*offset)) & 0xFF
                        if count != 0 and pixels[index*8+offset] == 0:
                            pixels[index*8 + offset] = int(delay_path[-3:])
                    index += 1
        print(pixels)
        print(np.amax(pixels) - np.amin(pixels))
        matrix = np.reshape(pixels, (14, 14))
        im = ax.imshow(matrix)

        ax.set_title("Heatmap du délais de réception")
        fig.tight_layout()
        plt.show()

main()
