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
    def __init__(self, filename, basePath, pixel_id, filter_lower_than=0.05):
        self.density_code, self.fine_by_coarse, self.min_fine_by_coarse, self.min_coarse = self.get_density_code(filename, basePath, pixel_id, filter_lower_than)
        self.number_of_coarse = len(self.fine_by_coarse)

        print(self.min_coarse)
        print(self.min_fine_by_coarse)
        ps_per_count = PERIOD_IN_PS / np.sum(self.density_code)
        time_per_code = self.density_code * ps_per_count
        self.ps_per_coarse = PERIOD_IN_PS / (np.sum(self.fine_by_coarse) / np.mean(self.fine_by_coarse[:-1]))   # Last coarse smaller so remove it from mean
        self.ideal_tf = np.cumsum(time_per_code)
        print(len(self.ideal_tf))
        self.global_bias = 0

        self.coarse_lookup_table = np.zeros(2**4)
        self.fine_slope_corr_lookup_table = np.zeros(2 ** 4)

        self.lin_reg = self._linear_regression_algorithm()
        self.med_reg = self._median_algorithm()
        self.bias_reg = self._linear_regression_algorithm(True, False)
        self.bias_slope_reg = self._linear_regression_algorithm(True, True)

    def get_density_code(self, filename, basePath, pixel_id, filter_lower_than):
        with h5py.File(filename, "r") as h:
            ds = h[basePath]
            coarse = np.array(ds['Coarse'], dtype='int64')
            fine = np.array(ds['Fine'], dtype='int64')
            addr = np.array(ds['Addr'], dtype='int64')

            addr_filter = (addr == pixel_id)
            if not addr_filter.any():
                raise Exception("It seems like the targeted pixel was disabled or didn't trigger")

            coarse = coarse[addr_filter]
            fine = fine[addr_filter]

            H, xedges, yedges = np.histogram2d(coarse, fine, [max(coarse), max(fine)],
                                               range=[[0, max(coarse)], [0, max(fine)]])
            # Filter out
            min_fine_by_coarse = np.argmax(~(H < (np.amax(H) * filter_lower_than)), axis=1)
            fine_by_coarse = np.sum(~(H < (np.amax(H) * filter_lower_than)), axis=1)
            fine_by_coarse = fine_by_coarse[fine_by_coarse != 0]
            density_code = H[~(H < (np.amax(H) * filter_lower_than))]
            return density_code, fine_by_coarse, min_fine_by_coarse, min(coarse)

    def get_ideal(self):
        return self.ideal_tf

    def get_linear(self):
        return self.lin_reg

    def get_median(self):
        return self.med_reg

    def get_biased_linear(self):
        return self.bias_reg

    def get_slope_corr_biased_linear(self):
        return self.bias_slope_reg

    def code_to_timestamp(self, coarse, fine):
        index = np.sum(self.fine_by_coarse[:coarse]) + fine
        return self.ideal_tf[index]

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

        coarse_mean_point = []
        avg_fine_by_coarse = np.mean(self.fine_by_coarse)
        for coarse in range(len(self.fine_by_coarse)):
            number_of_fine = self.fine_by_coarse[coarse]
            # Forget if only a few points
            if number_of_fine < 0.7 * avg_fine_by_coarse:
                offset += number_of_fine
                if coarse < 2:
                    self.min_coarse = coarse + 1
                continue
            current_data = self.ideal_tf[offset:offset+number_of_fine]
            offset += number_of_fine
            coarse_mean_point.append(np.mean(current_data))

        # First get coarse approx
        parameters = self._linear_regression(coarse_mean_point)
        a = parameters[0]
        b = parameters[1]
        self.global_bias = -b - a/2
        self.coarse_period = np.around(a * 8) / 8    # Apply the same resolution as in the chip

        offset=0
        for coarse in range(len(self.fine_by_coarse)):
            number_of_fine = self.fine_by_coarse[coarse]
            current_data = self.ideal_tf[offset:offset+number_of_fine] - self.coarse_period * coarse
            offset += number_of_fine
            #TODO
            #self.min_fine_by_coarse[coarse]
            parameters = self._linear_regression(current_data[1:-1])
            if parameters is None:
                continue
            a = parameters[0]
            b = parameters[1]
            slopes.append(a)
            bias.append(b)

        if not lookup_slope:
            self.fine_period = np.around(np.average(np.array(slopes)) * 16) / 16
            #self.global_bias += np.average(bias)
        else:
            self.fine_period = max(slopes)
            self._fill_correction_table_for_fine_slope(slopes)

        if lookup_bias:
            self._fill_lookup_table_coarse(self.ideal_tf)
            #self._fill_lookup_bias(bias)

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
               (self.min_fine_by_coarse[coarse] + fine) * (self.fine_period + self.fine_slope_corr_lookup_table[coarse]) + self.min_coarse * self.global_bias

    def _fill_lookup_bias(self, bias):
        for coarse in range(len(self.fine_by_coarse)):
            self.coarse_lookup_table[coarse] = round(bias[coarse])


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
            self.coarse_lookup_table[coarse] = min(round(average), 256)

    def _fill_correction_table_for_fine_slope(self, slopes):
        for i in range(len(slopes)):
            diff = slopes[i] - self.fine_period
            self.fine_slope_corr_lookup_table[i+1] = min(((np.around(diff)/16) * 16), 1)
