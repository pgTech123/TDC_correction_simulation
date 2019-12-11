import copy
import numpy as np

########################################################
# CONSTANTS
########################################################
NUMBER_OF_TDC = 256
########################################################


class TransferFunctionNoCorrections:
    def __init__(self, transfer_function_ideal):
        self.transfer_function_ideal = transfer_function_ideal
        self._compute_transfer_function_y()

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
        return coarse * 500 + fine

    #######################################################################
    #                      PRIVATE FUNCTIONS
    #       (These functions shouldn't be called from outside)
    #######################################################################

    def _compute_transfer_function_y(self):
        y_tf = []
        [x, raw_data] = self.transfer_function_ideal.get_transfer_functions_raw_data()
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
