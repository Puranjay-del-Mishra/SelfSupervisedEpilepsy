import random
import torch
import numpy as np

def mask_value(signal):
    x = np.sum(signal, axis=-1)
    cnt = 8192
    return x/cnt

def mask_signal(signal,mask_count,alpha):
    mask_cnt = random.randint(0,mask_count)
    mask_len = random.randint(1,signal.shape[1]*alpha)
    if mask_cnt == 0:
        return signal
    else:
        i = mask_cnt
        masks = []
        mask = -1
        while i!=0 and mask_len!=0:
            if i!=1:
                mask = random.randint(1,mask_len)
            else:
                mask = mask_len
            masks.append(mask)
            mask_len -= mask
            i -=1
        print(masks)
        val = 0
        for mask in masks:
            cnt = mask
            end = signal.shape[1]
            start = -1
            while start<0:
                end = random.randint(0,signal.shape[1])
                start = end-cnt
            print(start,end,cnt)
            val = mask_value(signal)
            val = np.resize(val,[23,end-start])
            signal[:,start:end] = val   #masking the EEG values of EEG to masking value at this step
        return signal

