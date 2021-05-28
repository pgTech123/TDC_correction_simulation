import math
import h5py
import numpy as np


def hdf52Timestamp(filename, basePath):
    with h5py.File(filename, "r") as h:
        ds = h[basePath]
        value = np.array(ds, dtype='int64')
        address = np.right_shift(np.bitwise_and(value, 0x1FF), 0)
        energy = np.right_shift(np.bitwise_and(value, 0x1FE00), 9)
        timestamp = np.right_shift(np.bitwise_and(value, 0x7FFFFFFFFE0000), 17)
        print(address)
        print(energy)
        print(timestamp)
        print(np.diff(timestamp).tolist())

        """
        addr_filter = (addr == pixel_id)
        if not addr_filter.any():
            raise Exception("It seems like the targeted pixel was disabled or didn't trigger")

        coarse = coarse[addr_filter]
        fine = fine[addr_filter]

        H, xedges, yedges = np.histogram2d(coarse, fine, [max(coarse), max(fine)],
                                           range=[[0, max(coarse) + 1], [0, max(fine) + 1]])
        # Filter out
        min_fine_by_coarse = np.argmax(~(H < (np.amax(H) * filter_lower_than)), axis=1)
        fine_by_coarse = np.sum(~(H < (np.amax(H) * filter_lower_than)), axis=1)
        fine_by_coarse = fine_by_coarse[fine_by_coarse != 0]
        density_code = H[~(H < (np.amax(H) * filter_lower_than))]
        return density_code, fine_by_coarse, min_fine_by_coarse, min(coarse)
        """


filename = "./data/ARR0/TDC_M0_NON_CORR_All-20210428-002531.hdf5"
basePath = "CHARTIER/ASIC0/TDC/M0/ALL_TDC_ACTIVE/PLL/FAST_255/SLOW_250/NON_CORR/EXT/ADDR_ALL/RAW"
hdf52Timestamp(filename, basePath)