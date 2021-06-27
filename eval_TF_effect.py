import numpy as np
import h5py
from tqdm import tqdm
from transferFunctionChip import TransferFunction
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

from scipy.stats import norm

"""
    Defining gaussian for curve fitting
"""
def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(- (x - mean)**2 / (2*standard_deviation ** 2))


def findMatchingTDCEvents(tdc1Num, tdc2Num, data):
    '''
    Finds the events with the same Global Counter (the same event) and returns the Coarse and Fine columns for both
    TDCs. They are ordered in matching pairs.
    :param tdc1Num: Number of the TDC # 1 to use
    :param tdc2Num: Number of the TDC # 2 to use
    :param data: Raw data, dont filter out the column names
    :return: the coarse,fine columns of all matched events for TDC#1 and TDC#2
    '''
    TDC1Data = data[(data["Addr"] == tdc1Num)]
    TDC2Data = data[(data["Addr"] == tdc2Num)]#[:TDC1Data.shape[0]]  # FIXME: Make the same length

    print(TDC1Data)
    print(TDC2Data)
    # Set columns Coarse and Fine
    data_type = np.dtype({'names': ['Coarse', 'Fine'], 'formats': ['u4', 'u4']})

    data1 = np.empty(0, dtype=data_type)
    data2 = np.empty(0, dtype=data_type)
    #to_keep = np.equal(TDC1Data['Global'], TDC2Data['Global'])

    print(len(TDC1Data))

    for i in tqdm(range(int(len(TDC1Data)/100))):
        for j in range(-50, 51, 1):
            if ((i+j) < 0) or ((i+j) >= len(TDC2Data['Global'])):
                continue
            if TDC1Data['Global'][i] == TDC2Data['Global'][i+j]:

                data1 = np.append(data1, np.array(TDC1Data[['Coarse', 'Fine']][i], dtype=data_type))
                data2 = np.append(data2, np.array(TDC2Data[['Coarse', 'Fine']][i+j], dtype=data_type))

    return data1, data2
    data1 = TDC1Data[['Coarse', 'Fine']]
    data2 = TDC2Data[['Coarse', 'Fine']]
    print(data1[to_keep])
    print(data2[to_keep])
    return data1[to_keep][:10000], data2[to_keep][:10000]


def curveFitting(diff, title):
    plt.figure()
    y, bins, patches = plt.hist(diff, bins=200, range=(-100, 100))
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

def main():
    filename = "./data/ARR0/NON_CORR_TEST_ALL-20210322-210304.hdf5"
    basePath = "CHARTIER/ASIC0/MO/TDC/NON_CORR/ALL/FAST_255/SLOW_250/ARRAY_0"
    filename = "./data/ARR0/EXP_NON_CORR_CALIB_COEF.hdf5"
    basePath = "CHARTIER/ASIC0/TDC/M0/ALL_TDC_ACTIVE/PLL/FAST_255/SLOW_250/NON_CORR/EXT/ADDR_ALL/RAW"

    with h5py.File(filename, "r") as h:
        ds = h[basePath]

        tdc1 = 4
        tdc2 = 8
        ev1, ev2 = findMatchingTDCEvents(tdc1, tdc2, ds)
        diff = []
        diff_offset = []
        diff_offset_both = []
        print(ev1)
        print(ev2)
        for (event1, event2) in zip(ev1, ev2):
            tf = TransferFunction()
            timestamp1 = tf.evaluate_ICSSHSR4(event1[1], event1[0], tdc1/4)
            timestamp2 = tf.evaluate_ICSSHSR4(event2[1], event2[0], tdc2/4)
            diff.append(timestamp2 - timestamp1)

        curveFitting(diff, "")

        #diff_ICSSHSR4_both = diff_ICSSHSR4_both[diff_ICSSHSR4_both<200]
        #plt.figure()
        #y, bins, patches = plt.hist(diff_ICSSHSR4_both, bins=30, range=(-150, 150))

        #curveFitting(diff_ICSSHSR4_both, "Same as ICSSHSR4")

        plt.show()


main()