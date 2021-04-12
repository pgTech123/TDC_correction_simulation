import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

tdc_y = 7
tdc_x = 7


with open('ratioLSB.pickle', 'rb') as f:
    ratios = pickle.load(f)
    ratio_arr = np.zeros((tdc_x, tdc_y))
    fig, ax = plt.subplots()


    for tdc_id in ratios:
        i_tdc_id = int(tdc_id)
        i = int(i_tdc_id/tdc_y)
        j = int(i_tdc_id % tdc_x)
        ratio_arr[i, j] = ratios[tdc_id]
        formated_text = round(float(ratio_arr[i, j]), 2)
        text = ax.text(j, i, formated_text, ha="center", va="center", color="w")

    im = ax.imshow(ratio_arr)

    ax.set_title("Heatmap de la meilleure correction par rapport au LSB")
    fig.tight_layout()
    plt.show()
