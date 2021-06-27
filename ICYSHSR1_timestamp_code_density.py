# coding=utf-8
import math
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline


# TDC_VISUALIZED = 2
TOTAL_PERIOD = 4000

def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(- (x - mean)**2 / (2*standard_deviation ** 2))


def curveFitting(diff, title):
    plt.figure()
    y, bins, patches = plt.hist(diff, bins=50, range=(0, 100))
    plt.title(title)
    plt.xlabel("Différence de temps")
    plt.ylabel("Nombre de compte")

    x = bins[:-1] + 0.5

    mean = np.mean(diff)
    sigma = np.std(diff)

    print("------------")
    print(title)
    print(mean)
    print(sigma)

    popt, pcov = curve_fit(gaussian, x, y, p0=[1, mean, sigma])
    y_gauss = gaussian(x, *popt)

    spline = UnivariateSpline(x, y_gauss - np.max(y_gauss) / 2, s=0)
    r1, r2 = spline.roots()  # find the roots
    print(r1)
    print(r2)
    print("FWHM: " + str(abs(r2-r1)))

    plt.figure()
    plt.plot(x, y, 'b+:', label='data')
    plt.plot(x, gaussian(x, *popt), 'ro:', label='fit')
    plt.legend()
    plt.title(title)
    plt.xlabel("Différence de temps")
    plt.ylabel("Nombre de compte")


def main():
    # ARR 0
    # Fichier Pascal
    filename = "./data/ARR0/TDC_M0_NON_CORR_TIME_All-20210604-140203.hdf5"
    filename = "./data/ARR0/TIME_ALL_CORR/TDC_M0_NON_CORR_TIME_All-20210519-140903.hdf5"
    #filename = "./data/ARR0/TIME_CORR_ICSSHSR5/TDC_M0_NON_CORR_TIME_All-20210519-154704.hdf5"
    #filename = "./data/ARR0/TIME_CORR_BIAS_ONLY/TDC_M0_NON_CORR_TIME_All-20210612-212231.hdf5"
    path = "CHARTIER/ASIC0/TDC/M0/ALL_TDC_ACTIVE/PLL/FAST_255/SLOW_250/NON_CORR/EXT/ADDR_ALL/RAW"

    with h5py.File(filename, "r") as h:
        packet = np.array(h[path])
        data = packet[packet != 0xAAAAAAAAAAAAAAAA]
        data = data[data != 0xAAAAAAABAAAAAAAB]
        data = data[data != 0]

        address = np.bitwise_and(data, 0x1FF)
        energy = np.bitwise_and(np.right_shift(data, 9), 0xFF)
        timestamp = np.bitwise_and(np.right_shift(data, 17), 0x7FFFFFFFFF)
        coarse = np.bitwise_and(np.right_shift(data, 48), 0xF)
        fine = np.bitwise_and(np.right_shift(data, 38), 0x3FF)

        diff = timestamp[address==4] + 50 - timestamp[address==8]
        diff = diff[diff <200]

        print(address)
        print(coarse)
        print(fine)
        print(energy)
        print(len(timestamp))
        print(diff)
        curveFitting(diff, "")

        plt.show()


main()
