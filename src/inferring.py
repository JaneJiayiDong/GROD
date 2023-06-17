
import torch
import numpy as np
import anndata
import scanpy as sc
from src.utils.misc import get_dataset_information, create_adata
import pandas as pd
### https://github.com/KrishnaswamyLab/MIOFlow/blob/250ec16d0ee0cb0c16db9a101db2deb5b3cc4542/MIOFlow/eval.py

def generate_plot_data(model, data, graph, n_bins=20):
    points = generate_points(model, data, graph)
    trajectories = generate_trajectories(model, data, graph, n_bins)
    return points, trajectories

def generate_points(model, data, graph, sample_data=None):
    x, y, x_t, y_t, x_mask, y_mask, t_id = data
    if sample_data == None:
        sample_data = y_t 
    generated, _, _ = model(x, x_t, sample_data, graph)
    return generated, t_id

def generate_trajectories(model, data, graph ,n_bins=5):
    _, _, _, y_t, _, _, _ = data
    sample_time = torch.linspace(0, 1, n_bins).to(y_t).view(1, -1) # torch.linspace(y_t.min(), y_t.max(), n_bins).to(y_t).view(1, -1)
    trajectories, t_id = generate_points(model, data, graph, sample_time)
    return trajectories, t_id

def infer_trajectory(train_iterator, model, graph, raw_data, cell_name_list, data_name, pred_data=None):
    if pred_data != None:
        adata = create_adata(pred_data, raw_data) 
        adata.obs_names = cell_name_list
        split_str, point, replace_dic, cls_num = get_dataset_information(data_name)
        t_list = adata.obs_names.to_series().apply(lambda x:x.split(split_str)[point])
        adata.obs['time'] = t_list
    
    startpoints_list = []
    trajectories_list = []
    ids_list = []
    for batch_idx, train_data in enumerate(train_iterator):
        trajectory, t_id = generate_trajectories(model, train_data, graph)
        startpoint = trajectory[:, 0, :, :].detach().cpu().numpy()
        trajectory = trajectory[:, 1:, :, :].detach().cpu().numpy() 
        t_id = t_id.detach().cpu().numpy()
        startpoints_list.append(startpoint)
        trajectories_list.append(trajectory)
        ids_list.append(t_id)
        
    # 整合数据
    trajectories = np.stack(trajectories_list, 0)
    startpoints = np.stack(startpoints_list, 0)
    ids = np.stack(ids_list, 0)
    trajectories = trajectories.reshape(-1, graph.shape[0])
    startpoints = startpoints.reshape(-1, graph.shape[0])
    ids = ids.reshape(-1, 1)
    
    tdata = create_adata(trajectories, raw_data) 
    # sdata = anndata.AnnData(X = startpoints)
    # split_str, point, replace_dic, cls_num = get_dataset_information(data_name)
    # t_list = pd.Series(cell_name_list).apply(lambda x:x.split(split_str)[point])
    tdata.obs['time'] = 'pred'
    # sdata.obs['time'] = t_list.values[ids.squeeze()]
    # sdata.obs['time'] = sdata.obs['time'].replace(replace_dic)
    # tdata = anndata.concat([tdata, sdata])
    
    # adata = anndata.concat([adata, tdata])
    # # 设置标签
    if pred_data != None:
        adata = anndata.concat([adata, tdata])
    else:
        adata = tdata
        
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.pl.umap(adata,
    color=['time'],
    wspace=0.4,
    legend_fontsize=14,
    save= 'trajectories.png',
    frameon=False)
