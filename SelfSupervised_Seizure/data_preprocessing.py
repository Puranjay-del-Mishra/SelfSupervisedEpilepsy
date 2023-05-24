import numpy
import pyedflib
import numpy as np
from pyedflib import highlevel
import pickle
import torch
import os
from masking_function import mask_signal
import faulthandler

normal_load_loc = 'C:/Users/puran/PycharmProjects/SelfSupervised_Seizure/Normal_edf_data'   #no data with epileptic activity
anomaly_load_loc = 'C:/Users/puran/PycharmProjects/SelfSupervised_Seizure/Anomalous_edf_data' # data with only epileptic activity

normal_save_loc = 'C:/Users/puran/PycharmProjects/SelfSupervised_Seizure/normal_ndarray_data'  # save location for data without epileptic activity
all_save_loc = 'C:/Users/puran\PycharmProjects/SelfSupervised_Seizure/all_ndarray_data'  #save location for all the data (with and without). This contains a dictionary of the input as well as the labels


def preprocess_data():
    normal_eeg_signals = 0
    anomaly_eeg_signals = 0

    flag0 = 0

    ndarray_list = []

    flag0 = -1

    for data in os.listdir(normal_load_loc):
        signals, signal_headers, header = highlevel.read_edf(os.path.join(normal_load_loc, data),
                                                             ch_names=['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3',
                                                                       'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4',
                                                                       'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8',
                                                                       'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9',
                                                                       'FT9-FT10', 'FT10-T8', 'T8-P8'])
        ndarray_list.append(signals)

    ndarray_list_2 = numpy.concatenate(ndarray_list,axis=-1)

    inp_len = 8192

    for i in range(1, int(ndarray_list_2.shape[1]/ inp_len)):
        print(i)
        array = ndarray_list_2[:, (i - 1) * inp_len:i * inp_len]
        masked_array = mask_signal(array, 3, 0.0625)
        Dict = {'array':array,'masked':masked_array}
        with open(os.path.join(normal_save_loc,f'{i}.pickle'), 'wb') as handle:
            pickle.dump(Dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # try:
        #     torch.save(temp_tensor, os.path.join(normal_save_loc, str(i) + '.pt'))
        # except: pass
        if i==1500:
            print('Breakpoint in the loop...')
            break
    print('Normal eeg signals saved as a tensors')
    flag0 = 0

    for data in os.listdir(anomaly_load_loc):
        signals, signal_headers, header = highlevel.read_edf(os.path.join(anomaly_load_loc, data),
                                                             ch_names=['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3',
                                                                       'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4',
                                                                       'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8',
                                                                       'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9',
                                                                       'FT9-FT10', 'FT10-T8', 'T8-P8'])
        torch_signal = torch.Tensor(signals)
        if flag0 == 0:
            anomaly_eeg_signals = torch_signal
            flag0 = 1
        else:
            torch.cat((anomaly_eeg_signals, torch_signal), dim=1)

    print('Anomaly eeg signals procured as a tensor')
    print(anomaly_eeg_signals.shape[1])

    for i in range(1, int(normal_eeg_signals.shape[1]/ inp_len)):
        temp_tensor = normal_eeg_signals[:, (i - 1) * inp_len:i * inp_len]
        temp_tensor = mask_signal(temp_tensor)
        temp_dict = {'input': temp_tensor, 'label': 0}
        try:
            torch.save(temp_dict, os.path.join(all_save_loc, str(i) + '.pt'))
        except Exception as e: print(e)

    for i in range(1, int(anomaly_eeg_signals.shape[1]/ inp_len)):
        temp_tensor = anomaly_eeg_signals[:, (i - 1) * inp_len:i * inp_len]
        temp_tensor = mask_signal(temp_tensor)
        temp_dict = {'input': temp_tensor, 'label': 1}
        try:
            torch.save(temp_dict, os.path.join(all_save_loc, str(i) + '.pt'))
        except Exception as e: print(e)


if __name__ == '__main__':
    # faulthandler.enable()
    preprocess_data()