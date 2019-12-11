import reader.reader as reader
import numpy as np


NUMBER_OF_TDC = 256


def read_tdc(filename, tdc, timestamp=True, energy=True, valid_data_only=True):
    timestamp_flag = 0
    energy_flag = 0
    valid_flag = 0
    
    if timestamp:
        timestamp_flag = 1
    if energy:
        energy_flag = 1
    if valid_data_only:
        valid_flag = 1
    print("Loading file. This might take a while.")
    return reader.read_tdc(filename, energy_flag, timestamp_flag, valid_flag, tdc)


def read_file(filename, timestamp=True, energy=True, valid_data_only=True):
    timestamp_flag = 0
    energy_flag = 0
    valid_flag = 0
    
    if timestamp:
        timestamp_flag = 1
    if energy:
        energy_flag = 1
    if valid_data_only:
        valid_flag = 1
    print("Loading file. This might take a while.")
    return reader.read_file(filename, energy_flag, timestamp_flag, valid_flag)


# tdc_data = output from read_tdc
def get_tdc_column(tdc_data, column):
    return [tdc_content[column] for tdc_content in tdc_data]


def get_histogram_raw(filename):
    print("Loading file. This might take a while.")
    return reader.get_histogram(filename)


def get_max_coarses(histogram):
    max_coarses = []
    for tdc_histogram in histogram:
        max_coarses.append(tdc_histogram[1][-1][0])
    return max_coarses


def get_max_fines(histogram):
    max_fines = []
    for tdc_histogram in histogram:
        cur_max_fine = 0
        for coarse in tdc_histogram[1]:
            if coarse[1][-1][0] > cur_max_fine:
                cur_max_fine = coarse[1][-1][0]
        max_fines.append(cur_max_fine)
    return max_fines


def get_histogram_np(filename):
    histogram_raw = get_histogram_raw(filename)
    hist_list = []
    max_fine_all = []
    max_coarse_all = []
    fine_count_per_coarse = []
    for i in range(NUMBER_OF_TDC):
        hist_list.append([])
        max_fine_all.append(0)
        max_coarse_all.append(0)
        fine_count_per_coarse.append([0])

    max_fines = get_max_fines(histogram_raw)
    max_coarses = get_max_coarses(histogram_raw)
    for h, max_fine, max_coarse in zip(histogram_raw, max_fines, max_coarses):
        # Create the array to store data
        address = h[0]
        if max_coarse == 0 or max_fine == 0:
            print("Invalid max_coarse: " + str(max_coarse) + " or max fine: " + str(max_fine) + " found")
            continue
        hist_list[address] = np.zeros(shape=((max_fine+1)*(max_coarse+1)))
        fine_count_per_coarse[address] = np.zeros(shape=(max_coarse+1))
        max_coarse_all[address] = max_coarse
        max_fine_all[address] = max_fine

        # Fill the array
        for coarse in h[1]:
            sum_fine_count = 0
            for fine in coarse[1]:
                hist_list[address][coarse[0]*max_fine + fine[0]] = fine[1]
                sum_fine_count += fine[1]
            fine_count_per_coarse[address][coarse[0]] = sum_fine_count
    return hist_list, max_coarse_all, max_fine_all, fine_count_per_coarse
