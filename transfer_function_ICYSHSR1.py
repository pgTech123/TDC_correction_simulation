import copy
import numpy as np

########################################################
# CONSTANTS
########################################################
NUMBER_OF_TDC = 256
########################################################


class TransferFunctionICYSHSR1:
    def __init__(self, transfer_function_ideal, algorithm="lookup_last_and_fine_correction"):
        self.transfer_function_ideal = transfer_function_ideal
        if algorithm == "lookup_coarse":
            self._lookup_table_coarse()
        elif algorithm == "lookup_and_fine_correction":
            self._lookup_table_coarse(with_correction_on_fine=True)
        elif algorithm == "lookup_last_and_fine_correction":
            self._lookup_table_coarse(with_correction_on_fine=True, lookup_table_for_last=True)
        else:
            raise Exception("Invalid algorithm")

    # Returns an array of size 256 containing true of false
    def get_mask_active_tdc(self):
        return self.transfer_function_ideal.get_mask_active_tdc()

    # Returns an array containing the ID of the active TDCs
    def get_active_tdc(self):
        return self.transfer_function_ideal.get_active_tdc()

    # Returns a copy of the transfer function raw data
    def get_transfer_functions_raw_data(self):
        # Make a copy of the list so that someone
        # outside this file cannot modify it.
        return [copy.deepcopy(self.x), copy.deepcopy(self.y_tf)]

    def evaluate(self, coarse, fine, tdc):
        #correction_lookup = 0
        #if self.lookup_table_last_code[tdc][0] == fine:
        #    correction_lookup = self.lookup_table_last_code[tdc][1]
        return coarse * self.coarse_period[tdc] + self.coarse_lookup_table[tdc, coarse] + \
               fine * (self.fine_period[tdc] + self.fine_slope_corr_lookup_table[tdc][coarse])

    #######################################################################
    #                      PRIVATE FUNCTIONS
    #       (These functions shouldn't be called from outside)
    #######################################################################
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

    def _get_best_slope_for_each_coarse(self, max_coarse, max_fine, y_tf):
        # Do a linear regression for every coarse, then average it
        slopes = []
        for coarse in range(max_coarse):
            number_of_fine = max_fine
            current_data = y_tf[coarse * number_of_fine:(coarse + 1) * number_of_fine]
            parameters = self._linear_regression(current_data)
            if parameters is None:
                continue
            a = parameters[0]
            b = parameters[1]
            slopes.append(a)
        return slopes

    def _lookup_table_coarse(self, with_correction_on_fine=False, lookup_table_for_last=True):
        self.active_tdc = self.transfer_function_ideal.get_active_tdc()
        max_fine = self.transfer_function_ideal.get_max_fine()
        max_coarse = self.transfer_function_ideal.get_max_coarse()
        [x, raw_data] = self.transfer_function_ideal.get_transfer_functions_raw_data()

        self.coarse_period = np.zeros(NUMBER_OF_TDC)
        self.lookup_table_last_code = np.zeros((NUMBER_OF_TDC, 2))
        self.coarse_lookup_table = np.zeros((NUMBER_OF_TDC, 2**4))
        self.fine_slope_corr_lookup_table = np.zeros((NUMBER_OF_TDC, 2 ** 4))
        self.fine_period = np.zeros(NUMBER_OF_TDC)
        for tdc in range(NUMBER_OF_TDC):
            if tdc not in self.active_tdc:
                continue
            # Do a linear regression for every coarse, then average it
            slopes = self._get_best_slope_for_each_coarse(max_coarse[tdc], max_fine[tdc], raw_data[tdc])
            if with_correction_on_fine:
                self.fine_period[tdc] = min(slopes)
                self._fill_correction_table_for_fine_slope(tdc, slopes)
            else:
                average_slope = np.around(np.average(np.array(slopes)) * 16) / 16
                self.fine_period[tdc] = average_slope

            parameters = self._linear_regression(raw_data[tdc])
            if parameters is None:
                continue
            a = parameters[0]
            b = parameters[1]
            self.coarse_period[tdc] = np.around(a*max_fine[tdc] * 8) / 8    # Apply the same resolution as in the chip
            self._fill_lookup_table_coarse(raw_data[tdc], tdc, max_fine, max_coarse)
            if lookup_table_for_last:
                self._fill_lookup_table_last_fine(raw_data[tdc], tdc, max_fine, max_coarse)

        self._compute_transfer_function_y(x)

    def _fill_lookup_table_coarse(self, y_tf, tdc, max_fine, max_coarse):
        if max_coarse[tdc] == 0:
            return
        for coarse in range(max_coarse[tdc]+1):
            coarse_len = max_fine[tdc]
            cur_coarse_data_ideal = y_tf[coarse*coarse_len:(coarse+1)*coarse_len]
            # Get average offset between approx and real
            difference = []
            for fine, ideal_value in zip(range(len(cur_coarse_data_ideal)), cur_coarse_data_ideal):
                difference.append(ideal_value - self.evaluate(coarse, fine, tdc))
            average = np.average(np.array(difference))
            #print(difference)
            self.coarse_lookup_table[tdc, coarse] = round(average)

    def _fill_lookup_table_last_fine(self, y_tf, tdc, max_fine, max_coarse):
        if max_coarse[tdc] == 0:
            return
        difference = []
        for coarse in range(max_coarse[tdc] + 1):
            coarse_len = max_fine[tdc]
            cur_coarse_data_ideal = y_tf[coarse * coarse_len:(coarse + 1) * coarse_len - 1]
            difference.append(cur_coarse_data_ideal[-1] - self.evaluate(coarse, max_fine[tdc], tdc))
        average = np.average(np.array(difference))
        self.lookup_table_last_code[tdc][0] = max_fine[tdc] - 1
        self.lookup_table_last_code[tdc][1] = np.around(-average)
        #print(self.lookup_table_last_code[tdc])



    def _fill_correction_table_for_fine_slope(self, tdc, slopes):
        for i in range(len(slopes)):
            diff = slopes[i] - self.fine_period[tdc]
            self.fine_slope_corr_lookup_table[tdc][i] = np.around(diff)

    def _compute_transfer_function_y(self, x):
        y_tf = []
        max_fine = self.transfer_function_ideal.get_max_fine()
        for tdc in range(NUMBER_OF_TDC):
            y_estimated = []
            for x_value in x[tdc]:
                coarse = int(x_value / max_fine[tdc])
                fine = int(x_value % max_fine[tdc])
                estimated_time = self.evaluate(coarse, fine, tdc)
                y_estimated.append(estimated_time)
            y_estimated_arr = np.array(y_estimated)
            y_tf.append(y_estimated_arr)
        self.x = x
        self.y_tf = y_tf
