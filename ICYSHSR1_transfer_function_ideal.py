import h5py
import numpy as np


########################################################
# CONSTANTS
########################################################
PERIOD_IN_PS = 4000.0
NUMBER_OF_TDC = 1
OUTLIER_THRESHOLD = 0.02
########################################################


class TransferFunctions:
    # Everything computation related should be called from the constructor
    def __init__(self, filename, basePath):
        self.tf_starts_at_origin = True
        self.density_code, self.fine_by_coarse = self.get_density_code(filename, basePath)
        self.number_of_coarse = len(self.fine_by_coarse)

        ps_per_count = PERIOD_IN_PS / np.sum(self.density_code)
        time_per_code = self.density_code * ps_per_count
        self.ps_per_coarse = PERIOD_IN_PS / (np.sum(self.fine_by_coarse) / np.mean(self.fine_by_coarse[:-1]))   # Last coarse smaller so remove it from mean
        self.ideal_tf = np.cumsum(time_per_code)

        self.coarse_lookup_table = np.zeros(2**4)
        self.fine_slope_corr_lookup_table = np.zeros(2 ** 4)

    def get_density_code(self, filename, basePath):
        with h5py.File(filename, "r") as h:
            ds = h[basePath]
            coarse = np.array(ds['Coarse'], dtype='int64')
            fine = np.array(ds['Fine'], dtype='int64')

            H, xedges, yedges = np.histogram2d(coarse, fine, [max(coarse), max(fine)],
                                               range=[[0, max(coarse)], [0, max(fine)]])
            # Filter out
            fine_by_coarse = np.sum(~(H < (np.amax(H) * 0.05)), axis=1)
            fine_by_coarse = fine_by_coarse[fine_by_coarse != 0]
            density_code = H[~(H < (np.amax(H) * 0.05))]
            return density_code, fine_by_coarse

    def get_ideal(self):
        return self.ideal_tf

    def get_linear(self):
        return self._linear_regression_algorithm()

    def get_median(self):
        return self._median_algorithm()

    def get_biased_linear(self):
        return self._linear_regression_algorithm(True, False)

    def get_slope_corr_biased_linear(self):
        return self._linear_regression_algorithm(True, True)

    #######################################################################
    #                      PRIVATE FUNCTIONS
    #       (These functions shouldn't be called from outside)
    #######################################################################
    @staticmethod
    def _get_median_step(y_tf):
        y = np.array(y_tf)
        diff = np.diff(y)
        median_step = np.median(diff)
        return median_step

    @staticmethod
    def _linear_regression(y_tf):
        number_of_points = len(y_tf)
        if number_of_points < 5:
            return None
        Y = np.transpose(np.array([y_tf]))
        x = np.arange(number_of_points)
        A_t = np.vstack((x, np.ones(number_of_points)))
        A = np.transpose(A_t)
        mult_A_t_Y = np.dot(A_t, Y)
        mult_A_t_A = np.dot(A_t, A)
        mult_A_t_A_inv = np.linalg.inv(mult_A_t_A)
        parameters = np.dot(mult_A_t_A_inv, mult_A_t_Y)
        return parameters[:, 0]

    @staticmethod
    def linear_regression_force_origin(y_tf):
        number_of_points = len(y_tf)
        if number_of_points < 5:
            return None
        Y = np.transpose(np.array([y_tf]))
        x = np.arange(number_of_points)
        a = np.sum(Y)/np.sum(x)
        return a

    def _median_algorithm(self):
        median_step = self._get_median_step(self.ideal_tf)
        self.fine_period = np.around(median_step)  # The height of the median step seems to be a good approximation.
        self.coarse_period = np.around(self.ps_per_coarse)
        return self._compute_transfer_function_y(range(len(self.ideal_tf)))

    def _linear_regression_algorithm(self, lookup_bias=False, lookup_slope=False):
        self.coarse_lookup_table = np.zeros(2**4)
        self.fine_slope_corr_lookup_table = np.zeros(2 ** 4)
        # Do a linear regression for every coarse, then average it
        slopes = []
        bias = []
        offset = 0
        for coarse in range(len(self.fine_by_coarse)):
            number_of_fine = self.fine_by_coarse[coarse]
            current_data = self.ideal_tf[offset:offset+number_of_fine]
            offset += number_of_fine
            parameters = self._linear_regression(current_data)
            if parameters is None:
                continue
            a = parameters[0]
            b = parameters[1]
            slopes.append(a)
            bias.append(b)

        if not lookup_slope:
            self.fine_period = np.around(np.average(np.array(slopes)) * 16) / 16
        else:
            self.fine_period = min(slopes)
            self._fill_correction_table_for_fine_slope(slopes)

        parameters = self._linear_regression(self.ideal_tf)
        a = parameters[0]
        b = parameters[1]
        self.coarse_period = np.around(a*np.mean(self.fine_by_coarse[:-1]) * 8) / 8    # Apply the same resolution as in the chip

        if lookup_bias:
            self._fill_lookup_table_coarse(self.ideal_tf)

        #a = self.linear_regression_force_origin(self.ideal_tf)
        #parameters = self._linear_regression(self.ideal_tf)
        #self.coarse_period = np.mean(self.fine_by_coarse[:-1]) * parameters[0] #np.around(a*max(self.fine_by_coarse))
        return self._compute_transfer_function_y(range(len(self.ideal_tf)))

    def _compute_transfer_function_y(self, x):
        y_estimated = []
        # Iterate all coarse
        coarse = 0
        for number_of_fine in self.fine_by_coarse:
            for fine in range(number_of_fine):
                estimated_time = self.evaluate(coarse, fine)
                y_estimated.append(estimated_time)
            coarse += 1
        return y_estimated

    def evaluate(self, coarse, fine):
        return coarse * self.coarse_period + self.coarse_lookup_table[coarse] + \
               fine * (self.fine_period + self.fine_slope_corr_lookup_table[coarse])

    def _fill_lookup_table_coarse(self, y_tf):
        offset = 0
        for coarse in range(len(self.fine_by_coarse)):
            cur_coarse_data_ideal = y_tf[offset:offset+self.fine_by_coarse[coarse]]
            offset += self.fine_by_coarse[coarse]
            # Get average offset between approx and real
            difference = []
            for fine, ideal_value in zip(range(len(cur_coarse_data_ideal)), cur_coarse_data_ideal):
                difference.append(ideal_value - self.evaluate(coarse, fine))
            average = np.average(np.array(difference))
            #print(difference)
            self.coarse_lookup_table[coarse] = round(average)

    def _fill_correction_table_for_fine_slope(self, slopes):
        for i in range(len(slopes)):
            diff = slopes[i] - self.fine_period
            self.fine_slope_corr_lookup_table[i] = np.around(diff)


