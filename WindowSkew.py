# coding=utf-8
import math
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    fig, ax = plt.subplots()
    filename = "./data/Window/Window_Size_Experiment-20210609-150059.hdf5"
    path1 = "CHARTIER/ASIC0/TDC/M0/ALL_TDC_ACTIVE/PLL/FAST_"
    path2 = "/SLOW_"
    path3 = "/WINDOW_LENGTH/EXT/ADDR_ALL/RAW"

    side = 14
    window_length = 128
    # Actual delays caracterized for the delay line
    delay_line = [18.39, 22.09, 48.31, 87.83, 162.73, 308.62, 612.52, 1229.95, 2453.47, 4877.48, 9764.98]
    pixels = np.zeros(side*side)
    pixels_del = np.zeros(side*side)
    with h5py.File(filename, "r") as h:
        path = path1 + str(window_length)
        delays = h[path].keys()
        for delay_path in delays:
            index = 0
            path = path1 + str(window_length) + "/" + delay_path + path3
            data = np.array(h[path])
            tmp = data[data != 0xAAAAAAAAAAAAAAAA]
            tmp = tmp[tmp != 0xAAAAAAABAAAAAAAB]
            tmp = tmp[tmp != 0]
            tmp = tmp[tmp != 327680]
            tmp = tmp[tmp != 327681]
            if len(tmp) == 0:
                continue
            for packet in data:
                if packet == 0xAAAAAAAAAAAAAAAA:
                    index = 0
                elif packet == 0xAAAAAAABAAAAAAAB:
                    index = 0
                elif packet == 327680:
                    pass
                elif packet == 327681:
                    pass
                else:
                    for offset in range(8):
                        count = (int(packet) >> (8*offset)) & 0xFF
                        if count != 0 and pixels[index*8+offset] == 0:
                            pixels[index*8 + offset] = int(delay_path[-3:])
                            del_bits = f'{int(delay_path[-3:]):010b}'
                            real_delay = lambda x: sum([int(str(x)[i])*delay_line[10-i] for i in range(10)])
                            pixels_del[index * 8 + offset] = real_delay(del_bits)
                    index += 1
        print(pixels)
        print(pixels_del)
        print(np.amax(pixels) - np.amin(pixels))
        print(np.amax(pixels_del) - np.amin(pixels_del))
        matrix = np.reshape(pixels, (side, side))
        matrix_del = np.reshape(pixels_del, (side, side))
        im = ax.imshow(matrix_del)

        ax.set_title("Délais entre la réception du signal de fenêtre pour les pixels de la matrice 8 x 8")
        ax = sns.heatmap(matrix_del- np.amin(pixels_del))

        fig.tight_layout()
        plt.show()

main()
