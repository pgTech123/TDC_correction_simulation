import copy
import numpy as np
import reader.reader_wrapper as reader


########################################################
# CONSTANTS
########################################################
PERIOD_IN_PS = 4000.0
NUMBER_OF_TDC = 256
OUTLIER_THRESHOLD = 0.02
########################################################


class TransferFunctionIdeal:
    # Everything computation related should be called from the constructor
    def __init__(self, filename):
        histogram_np, max_coarses, self.max_fines, self.fine_count_per_coarse_raw = reader.get_histogram_np(filename)
        self.histograms = self.filter_histogram(histogram_np)
        self.compute_transfer_function()


    # Returns an array of size 256 containing true of false
    def get_mask_active_tdc(self):
        active_tdcs = []
        for tdc in range(NUMBER_OF_TDC):
            if len(self.histograms[tdc]) > 0:
                active_tdcs.append(True)
            else:
                active_tdcs.append(False)
        return active_tdcs


    # Returns an array containing the ID of the active TDCs
    def get_active_tdc(self):
        tdc_found = []
        for tdc in range(NUMBER_OF_TDC):
            if len(self.histograms[tdc]) > 0:
                tdc_found.append(tdc)
        return tdc_found


    # Returns a copy of the max fine found for each addresses
    def get_max_fine(self):
        # Make a copy of the list so that someone
        # outside this file cannot modify it.
        return list(self.max_fines)


    # Returns a copy of the max fine found for each addresses
    def get_max_coarse(self):
        # Make a copy of the list so that someone
        # outside this file cannot modify it.
        return list(self.max_coarse)


    # Returns a copy of the transfer function raw data
    def get_transfer_functions_raw_data(self):
        # Make a copy of the list so that someone
        # outside this file cannot modify it.
        return [copy.deepcopy(self.x), copy.deepcopy(self.y_tf)]


    # Returns a copy of the inl raw data
    def get_inl_data(self):
        # Make a copy of the list so that someone
        # outside this file cannot modify it.
        return [copy.deepcopy(self.x), copy.deepcopy(self.inl)]


    # Returns a copy of the dnl raw data
    def get_dnl_data(self):
        # Make a copy of the list so that someone
        # outside this file cannot modify it.
        return [copy.deepcopy(self.x), copy.deepcopy(self.dnl)]


    # Returns a copy of the histograms
    def get_histograms(self):
        histograms = []
        for hist in self.histograms:
            if len(hist) > 0:
                bins = np.arange(len(hist)+1)
                histograms.append(np.array([hist,bins]))
            else:
                bins = np.array([0,1])
                empty_hist = np.array([0])
                histograms.append(np.array([empty_hist, bins]))
        return np.array(histograms)


    # Returns a copy of the fine_count_per_coarse
    def get_fine_count_per_coarse(self):
        # Make a copy of the list so that someone
        # outside this file cannot modify it.
        return copy.deepcopy(self.fine_count_per_coarse)


    # Returns a copy of the ps_per_coarse
    def get_ps_per_coarse(self):
        # Make a copy of the list so that someone
        # outside this file cannot modify it.
        return list(self.ps_per_coarse)


    #######################################################################
    #                      PRIVATE FUNCTIONS
    #       (These functions shouldn't be called from outside)
    #######################################################################
    def is_outlier(self, histogram):
        hist_max = np.amax(histogram)    # Find the greatest value in the histogram
        outliers = np.zeros(len(histogram), dtype=np.bool_)
        # Starting from the end, if the value is smaller than threshold*hist_max,
        # remove it. Stop doing that as soon as we find a value that doesn't need to
        # be removed. We don't want to remove values in the middle of the histogram.
        for i in range(len(histogram)):
            if histogram[-i] == 0:
                outliers[-i] = True
            elif histogram[-i] < OUTLIER_THRESHOLD * hist_max:
                outliers[-i] = True
            else:
                break  # If not at the very end, stop
        return outliers


    def filter_histogram(self, histogram):
        # Remove unused bins and data
        filtered_histograms = []
        for hist in histogram:
            if len(hist) > 0:
                outliers = self.is_outlier(hist)
                filtered_hist = hist[~outliers]
            else:
                filtered_hist = []
            # Make sure that we have an empty histogram if no TDCs were triggered
            if len(filtered_hist) == 0:
                filtered_hist = []
            filtered_histograms.append(filtered_hist)
        return filtered_histograms



    def adjust_max_coarse(self, histograms_corrected, fines_max):
        max_coarse = []
        for histogram, fine_max in zip(histograms_corrected, fines_max):
            # Case where TDC didn't trigger
            if fine_max == 0:
                max_coarse.append(0)
            else:
                # The last value of the bins (histogram[1]) represent the highest value received
                # Since the bins begin at -0.5, and end 0.5 above the last value received, retrieve 0.5
                max_value = len(histogram)
                max_coarse.append(int(max_value / fine_max))    # Coarse = Total / fine_max
        return np.array(max_coarse)


    def calcDNL_INL_TF(self, tot_Hist, fine_per_coarse, coarse_period):
        res_fine = coarse_period / fine_per_coarse

        dnl = ((tot_Hist - np.mean(tot_Hist)) / np.mean(tot_Hist)) * res_fine
        inl = np.cumsum(dnl)
        tf = np.cumsum(res_fine + dnl)
        x = list(range(len(dnl)))

        return dnl, inl, tf, x


    def compute_transfer_function(self, debug=True):
        coarse_max = self.adjust_max_coarse(self.histograms, self.max_fines)
        if debug:
            print("coarse_max : " + str(coarse_max))
            print("fine_max : " + str(self.max_fines))

        # Keep only the fine coarse lower than coarse_max
        fine_count_per_coarse = [self.fine_count_per_coarse_raw[tdc][:coarse_max[tdc] + 1] for tdc in range(NUMBER_OF_TDC)]
        ps_per_coarse = [PERIOD_IN_PS / (np.sum(fcpc)/np.mean(fcpc[:-1])) for (cm, fcpc) in zip(coarse_max, fine_count_per_coarse)]

        self.x = []
        self.dnl = []
        self.inl = []
        self.y_tf = []
        for (h, fm, ppc) in zip(self.histograms, self.max_fines, ps_per_coarse):
            dit_data = self.calcDNL_INL_TF(h, fm, ppc)
            self.x.append(dit_data[3])
            self.dnl.append(dit_data[0])
            self.inl.append(dit_data[1])
            self.y_tf.append(dit_data[2])

        self.max_coarse = coarse_max
        self.fine_count_per_coarse = fine_count_per_coarse
        self.ps_per_coarse = ps_per_coarse


