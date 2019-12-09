

class TransferFunctionICSSHSR4:
    def __init__(self, transfer_function_ideal, algorithm="linear_regression"):
        self.active_tdc = transfer_function_ideal.get_active_tdc()
        self.max_fine = transfer_function_ideal.get_max_fine()
        ps_per_coarse = transfer_function_ideal.get_ps_per_coarse()
        [x, raw_data] = transfer_function_ideal.get_transfer_functions_raw_data()

