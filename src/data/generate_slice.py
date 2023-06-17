import numpy as np
import torch
from einops import rearrange


def generate_slice(data, mask, time, input_step, pred_step):
    x_offsets = np.sort(np.concatenate((np.arange(-(input_step - 1), 1, 1),))) 
    y_offsets = np.sort(np.arange(1, (pred_step + 1), 1))

    x, y, x_t, y_t, x_mask, y_mask, t_id = generate_seq2seq_io_data(
        data, 
        time,
        mask,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
    )

    return x, y, x_t, y_t, x_mask, y_mask, t_id

def generate_seq2seq_io_data(
        data, time, mask, x_offsets, y_offsets, scaler=None
):
    """
    Generate samples from
    :param df: samples * nodes
    :param tf: samples * 1
    :param mask: samples * nodes
    :param x_offsets: 
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes, D = data.shape    


    x, y = [], []
    x_t, y_t = [], []
    x_mask, y_mask = [], []
    t_id = []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
        x_t.append(time[t + x_offsets, ...])
        y_t.append(time[t + y_offsets, ...])
        x_mask.append(mask[t + x_offsets, ...])
        y_mask.append(mask[t + y_offsets, ...])
        t_id.append(t)
    x = torch.stack(x, axis=0)
    y = torch.stack(y, axis=0)
    x_t = torch.stack(x_t, axis=0)
    y_t = torch.stack(y_t, axis=0)
    x_mask = torch.stack(x_mask, axis=0)
    y_mask = torch.stack(y_mask, axis=0)
    t_id = torch.LongTensor(t_id)
    return x, y, x_t, y_t, x_mask, y_mask, t_id
