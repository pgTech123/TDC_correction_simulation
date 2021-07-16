# coding=utf-8
import math
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


array = 0

side = 14
if array:
    side = 8
total_pixels = side*side

# Actual delays caracterized for the delay line
#delay_line = [18.39, 22.09, 48.31, 87.83, 162.73, 308.62, 612.52, 1229.95, 2453.47, 4877.48]
delay_line = [10, 15, 35, 70, 145, 290, 575, 1150, 2300, 4610]


def read_counts(data):
    tmp = data[data != 0xAAAAAAAAAAAAAAAA]
    tmp = tmp[tmp != 0xAAAAAAABAAAAAAAB]
    tmp = tmp[tmp != 327680]
    tmp = tmp[tmp != 327681]
    tot_entry = (int(tmp.shape[0]/(total_pixels/8)))
    counts = np.zeros((tot_entry, total_pixels))
    begun = False
    entry_index = 0
    for packet in data:
        if packet == 0xAAAAAAAAAAAAAAAA:
            index = 0
            begun = True
        elif not begun:
            continue
        elif packet == 0xAAAAAAABAAAAAAAB:
            index = 0
            entry_index += 1
        elif packet == 327680:
            pass
        elif packet == 327681:
            pass
        else:
            for offset in range(8):
                pixel_index = index*8+offset
                if pixel_index >= total_pixels:
                    continue
                count = (int(packet) >> (8 * offset)) & 0xFF
                if entry_index < tot_entry:
                    counts[entry_index, pixel_index] = count
            index += 1
    # Remove zero counts
    return counts[~np.all(counts == 0, axis=1)]


def get_average_counts(counts):
    return np.average(counts, axis=0)


def get_window_len(avg_counts, x, approx_with_x=True, one_threshold=False, flip=False):
    threshold = np.amax(avg_counts, axis=0)/20
    above_threshold = avg_counts > threshold
    if flip:
        above_threshold = np.flip(above_threshold, axis=0)
    # Index non zero
    if approx_with_x:
        start_index = np.argmax(above_threshold, axis=0)
        start_time = x[start_index]
        if one_threshold:
            return start_time
        end_index = above_threshold.shape[0] - np.argmax(np.flip(above_threshold, axis=0), axis=0)
        # Convert to time
        end_time = x[end_index]
        win_len = end_time - start_time
        return win_len
    else:
        win_len = np.count_nonzero(above_threshold, axis=0)
        return win_len


def delay_path_to_actual_delay(path_del):
    delay = int(str(path_del).replace('SLOW_', ''))
    delay_pos = abs(delay)
    del_bits = f'{delay_pos:010b}'
    real_delay = lambda x: sum([int(str(x)[i]) * delay_line[9 - i] for i in range(10)])
    actual_delay = real_delay(del_bits)
    if delay < 0:
        actual_delay = -actual_delay
    return actual_delay


def read_window_len(filename, max_win=9, one_threshold=False, flip=False):
    path1 = "CHARTIER/ASIC0/TDC/M0/ALL_TDC_ACTIVE/PLL"
    #path1 = "CHARTIER/ASIC0/TDC/M1/ALL_TDC_ACTIVE/PLL"
    path2 = "/SLOW_"
    path3 = "/WINDOW_LENGTH/EXT/ADDR_ALL/RAW"
    window_graph = []
    start_delay = []
    with h5py.File(filename, "r") as h:
        # Remove the FAST_ to get the window len
        objList = list(h[path1].keys())
        win_len = list(map(lambda x: int(str(x).replace('FAST_', '')), objList))
        win_len.sort()
        window_code = []
        for window in win_len[:max_win]:
            print("Processing window "+ str(window))
            path = path1 + "/FAST_" + str(window)
            delays = h[path].keys()
            graph = []
            x = []
            for delay_path in delays:
                actual_delay = delay_path_to_actual_delay(delay_path)
                # Uncomment when 2 windows in the data
                # if actual_delay < 0:
                #     continue
                path = path1 + "/FAST_" + str(window) + "/" + delay_path + path3
                data = np.array(h[path])
                # Quickly ignore empty frames
                tmp = data[data != 0xAAAAAAAAAAAAAAAA]
                tmp = tmp[tmp != 0xAAAAAAABAAAAAAAB]
                tmp = tmp[tmp != 0]
                tmp = tmp[tmp != 327680]
                tmp = tmp[tmp != 327681]
                if len(tmp) == 0:
                    continue

                counts = read_counts(data)
                avg = get_average_counts(counts)
                graph.append(avg)

                x.append(actual_delay)

            if len(graph) == 0:
                print("UNABLE TO PROCESS WINDOW " + str(window))
                window_graph.append(0)
                window_code.append(window)
                continue
            avg_count = np.array(graph)
            actual_window_len = get_window_len(avg_count, np.array(x), one_threshold=one_threshold, flip=flip)
            window_graph.append(np.mean(actual_window_len))
            window_code.append(window)
            start_delay.append(np.array(actual_window_len))
        return np.array(window_code), np.array(window_graph), np.array(start_delay)

def window_shape_graph():
    fig, ax = plt.subplots()
    filename = "./data/Window/GOOD_ALL_127-128-Window_Size_Experiment-20210609-150059.hdf5"
    path1 = "CHARTIER/ASIC0/TDC/M0/ALL_TDC_ACTIVE/PLL/FAST_"
    #path1 = "CHARTIER/ASIC0/TDC/M1/ALL_TDC_ACTIVE/PLL/FAST_"
    path2 = "/SLOW_"
    path3 = "/WINDOW_LENGTH/EXT/ADDR_ALL/RAW"

    window_length = 127

    pixels = np.zeros(side*side)
    pixels_del_begin = np.zeros(side*side)
    pixels_del_end = np.zeros(side*side)
    with h5py.File(filename, "r") as h:
        path = path1 + str(window_length)
        delays = h[path].keys()
        graph = []
        x = []
        for delay_path in delays:
            actual_delay = delay_path_to_actual_delay(delay_path)
            # Uncomment when multiple windows
            # if actual_delay < 0:
            #     continue
            path = path1 + str(window_length) + "/" + delay_path + path3
            data = np.array(h[path])
            # Quickly ignore empty frames
            tmp = data[data != 0xAAAAAAAAAAAAAAAA]
            tmp = tmp[tmp != 0xAAAAAAABAAAAAAAB]
            tmp = tmp[tmp != 0]
            tmp = tmp[tmp != 327680]
            tmp = tmp[tmp != 327681]
            if len(tmp) == 0:
                continue

            counts = read_counts(data)
            avg = get_average_counts(counts)
            graph.append(avg)
            x.append(actual_delay)

        #matrix = np.reshape(pixels, (side, side))
        #matrix_del = np.reshape(pixels_del_begin, (side, side))
        #im = ax.imshow(matrix_del)
        avg_count = np.array(graph)
        win_len = get_window_len(avg_count, np.array(x), False)
        print(np.mean(win_len))
        #ax.set_title("Délais entre la réception du signal de fenêtre pour les pixels de la matrice 8 x 8")
        #ax = sns.heatmap(matrix_del- np.amin(pixels_del_begin))
        plt.plot(np.array(x), avg_count[:, 0])
        #plt.title("Nombre de comptes selon le délais appliqué à la fenêtre")
        plt.xlabel('Délais (ps)')
        plt.ylabel('Nombre de comptes')
        #fig.tight_layout()
        plt.show()


def window_length_graph():
    filename = "./data/Window/GOOD_ALL_127-128-Window_Size_Experiment-20210609-150059.hdf5"
    win_code, actual_len, _ = read_window_len(filename, 71)
    print(actual_len)
    print(actual_len.shape)
    plt.plot(win_code, actual_len)
    #plt.title("Durée réelle de la fenêtre en fonction du code.")
    plt.xlabel('Code de durée de la fenêtre')
    plt.ylabel('Durée réelle de la fenêtre (ps)')
    plt.show()


def get_threshold_value():
    """Used for long window"""
    filename = "./data/Window/GOOD-189-228-Window_Size_Experiment-20210612-222122.hdf5"
    win_code, actual_len, _ = read_window_len(filename, 9, one_threshold=True, flip=True)
    print(actual_len)
    print(actual_len - 1268)


def get_skew_graph():
    filename = "./data/Window/GOOD_1-23_Window_Size_Experiment-20210528-011951.hdf5"
    #filename = "./data/Window/Skew_petite_matrice_Window_Size_Experiment-20210603-022807.hdf5"
    _, _, start_del = read_window_len(filename, 23, one_threshold=True, flip=True)
    skew = []
    for delay in start_del:
        skew.append(delay - min(delay))

    mean_skew = np.mean(np.array(skew), axis=0)
    fig, ax = plt.subplots()

    matrix = np.reshape(mean_skew, (side, side))
    im = ax.imshow(matrix)

    #ax.set_title("D/callage  entre la réception du signal de fenêtre pour les pixels de la matrice 8 x 8")
    ax = sns.heatmap(matrix)

    fig.tight_layout()
    plt.show()


#window_shape_graph()
#window_length_graph()
#get_threshold_value()
get_skew_graph()