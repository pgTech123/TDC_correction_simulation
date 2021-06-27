import pickle


class TransferFunction:
    def __init__(self):
        pass

    def evaluate(self, fine, coarse, tdc):
        with open('20may_corr_coef_lin_bias_slope.pickle', 'rb') as f:
            coef = pickle.load(f)[tdc]
            #print(coef)
            coarse_time = coef[0]
            fine_time = coef[1]
            offset = coef[2]
            slope = coef[3]
            return coarse * coarse_time + offset[coarse] + fine * (fine_time + slope[coarse])

    def evaluate_bias_only(self, fine, coarse, tdc):
        with open('20may_corr_coef_lin_bias.pickle', 'rb') as f:
            coef = pickle.load(f)[tdc]
            coarse_time = coef[0]
            fine_time = coef[1]
            offset = coef[2]
            return coarse * coarse_time + offset[coarse] + fine * fine_time

    def evaluate_ICSSHSR4(self, fine, coarse, tdc):
        with open('20may_corr_coef_lin.pickle', 'rb') as f:
            coef = pickle.load(f)[tdc]
            coarse_time = coef[0]
            fine_time = coef[1]
            return coarse * coarse_time + fine * fine_time

