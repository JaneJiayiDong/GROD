import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import sys
import torch
import random
from scipy.sparse import linalg
import copy
import pandas as pd

class DataTimeLoader(object):
    def __init__(self, xs, ys, xs_t, ys_t, xs_mask, ys_mask, ts_id, batch_size, pad_with_last_sample=False, shuffle=False):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = xs[-1:].repeat([num_padding, 1, 1, 1])
            y_padding = ys[-1:].repeat([num_padding, 1, 1, 1])
            x_t_padding = xs_t[-1:].repeat([num_padding, 1, 1])
            y_t_padding = ys_t[-1:].repeat([num_padding, 1, 1])
            x_mask_padding = xs_mask[-1:].repeat([num_padding, 1, 1, 1])
            y_mask_padding = ys_mask[-1:].repeat([num_padding, 1, 1, 1])
            t_id_padding = ts_id[-1:].repeat([num_padding, 1])
            xs = torch.concatenate([xs, x_padding], axis=0)
            ys = torch.concatenate([ys, y_padding], axis=0)
            xs_t = torch.concatenate([xs_t, x_t_padding], axis=0)
            ys_t = torch.concatenate([ys_t, y_t_padding], axis=0)
            xs_mask = torch.concatenate([xs_mask, x_mask_padding], axis=0)
            ys_mask = torch.concatenate([ys_mask, y_mask_padding], axis=0)
            ts_id = torch.concatenate([ts_id, t_id_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size).tolist()
            xs, ys = xs[permutation], ys[permutation]
            xs_t, ys_t = xs_t[permutation], ys_t[permutation]
            xs_mask, ys_mask = xs_mask[permutation], ys_mask[permutation]
            ts_id = ts_id[permutation]
        self.xs = xs
        self.ys = ys
        self.xs_t = xs_t
        self.ys_t = ys_t
        self.xs_mask = xs_mask
        self.ys_mask = ys_mask
        self.ts_id = ts_id

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                x_t_i = self.xs_t[start_ind: end_ind, ...]
                y_t_i = self.ys_t[start_ind: end_ind, ...]  
                x_mask_i = self.xs_mask[start_ind: end_ind, ...]
                y_mask_i = self.ys_mask[start_ind: end_ind, ...]
                t_id_i = self.ts_id[start_ind: end_ind, ...]
                yield (x_i, y_i, x_t_i, y_t_i, x_mask_i, y_mask_i, t_id_i)
                self.current_ind += 1

        return _wrapper()


def split_dataset(dataset):
    x, y, x_t, y_t, x_mask, y_mask, t_id = dataset

    num_samples = x.shape[0]
    num_test = round(num_samples * 0)
    num_train = num_samples
    num_val = num_samples - num_test - num_train
    x_train, y_train, x_t_train, y_t_train, x_mask_train, y_mask_train, t_id_train  = (
        x[:num_train], y[:num_train], 
        x_t[:num_train], y_t[:num_train],
        x_mask[:num_train], y_mask[:num_train], t_id[:num_train]
    )
    train_data = {"x": x_train, 
                  "y": y_train,
                  "x_t": x_t_train,
                  "y_t": y_t_train,
                  "x_mask": x_mask_train,
                  "y_mask": y_mask_train,
                  "t_id": t_id_train}
    x_val, y_val, x_t_val, y_t_val, x_mask_val, y_mask_val, t_id_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
        x_t[num_train: num_train + num_val],
        y_t[num_train: num_train + num_val],
        x_mask[num_train: num_train + num_val],
        y_mask[num_train: num_train + num_val],
        t_id[num_train: num_train + num_val],
    )
    val_data = {"x": x_val, 
                "y": y_val,
                "x_t": x_t_val,
                "y_t": y_t_val,
                "x_mask": x_mask_val,
                "y_mask": y_mask_val,
                "t_id": t_id_val}
    x_test, y_test, x_t_test, y_t_test, x_mask_test, y_mask_test, t_id_test = (
        x[-num_test:], y[-num_test:], 
        x_t[-num_test:], y_t[-num_test:], 
        x_mask[-num_test:], y_mask[-num_test:],
        t_id[-num_test:],
    )
    test_data = {"x": x_test, 
            "y": y_test,
            "x_t": x_t_test,
            "y_t": y_t_test,
            "x_mask": x_mask_test,
            "y_mask": y_mask_test,
            "t_id": t_id_test}
    return train_data, val_data, test_data

def load_dataset(train_data, val_data, test_data, batch_size):
    # Data format
    data = {}

    data['train_loader'] = DataTimeLoader(train_data['x'], train_data['y'],train_data['x_t'], train_data['y_t'],
                                        train_data['x_mask'], train_data['y_mask'], train_data['t_id'], batch_size, shuffle=True)
    
    data['val_loader'] = DataTimeLoader(val_data['x'], val_data['y'], val_data['x_t'], val_data['y_t'],
                                        val_data['x_mask'], val_data['y_mask'], val_data['t_id'], batch_size, shuffle=False)
    
    data['test_loader'] = DataTimeLoader(test_data['x'], test_data['y'], test_data['x_t'], test_data['y_t'],
                                         test_data['x_mask'], test_data['y_mask'], test_data['t_id'], batch_size, shuffle=False)

    return data

