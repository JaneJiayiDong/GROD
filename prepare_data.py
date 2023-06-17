
import numpy as np
import pandas as pd
import os
from src.data.data_interpolate import interp_multivar_data

def prepare_real_data(opt, task):
    data_directory = opt.data_dir
    gene_expression_path = data_directory + 'ExpressionData.csv'
    pseudo_time_path = data_directory + 'PseudoTime.csv'

    sorted_gene_expression_path = data_directory + 'SortedExpressionData.csv'
    sorted_pseudo_time_path = data_directory + 'SortedPseudoTime.csv'

    # data
    gene_expr = pd.read_csv(gene_expression_path, index_col=0, header=0)
    gene_expr = gene_expr.T
    gene_name = list(gene_expr.columns)

    # time
    pseudo_time = pd.read_csv(pseudo_time_path, index_col=0, header=0)
    pseudo_time.drop_duplicates(subset=['PseudoTime'], keep='first', inplace=True)
    
    # sort data by time
    gene_expr = gene_expr.merge(pseudo_time[['PseudoTime']], left_index=True, right_index=True)
    gene_expr = gene_expr.sort_values('PseudoTime', ascending=True)
    cell_name = list(gene_expr.index)
    
    # cell_num
    if opt.cell_num < gene_expr.shape[0]:
        cell_num = min(opt.cell_num, gene_expr.shape[0])
        gene_expr = gene_expr.sample(n = cell_num)
        gene_expr = gene_expr.sort_values('PseudoTime', ascending=True)
        cell_name = list(gene_expr.index)
    
    # data and time
    sorted_pseudo_time = gene_expr['PseudoTime']
    gene_expr = gene_expr.drop(['PseudoTime'], axis=1)

    # save results
    sorted_pseudo_time.to_csv(sorted_pseudo_time_path)
    gene_expr.to_csv(sorted_gene_expression_path)

    # mask
    mask = (gene_expr.values != 0).astype(float)

    # normalization
    raw_data = gene_expr.values
    gene_expr = gene_expr.values
    
    if task == 'non_celltype_GRN':
        gene_expr = (gene_expr - gene_expr.mean(axis=0)) / gene_expr.std(axis=0)
    elif task == 'celltype_GRN' or task == 'imputation':
        means = []
        stds = []
        for i in range(gene_expr.shape[1]):  
            tmp = gene_expr[:, i]
            means.append(tmp[tmp != 0].mean())
            stds.append(tmp[tmp != 0].std())
        means = np.array(means)
        stds = np.array(stds)
        stds[np.isnan(stds)] = 0
        stds[np.isinf(stds)] = 0
        gene_expr = (gene_expr - means) / (stds)
        gene_expr[np.isnan(gene_expr)] = 0
        gene_expr[np.isinf(gene_expr)] = 0
        gene_expr = np.maximum(gene_expr, -15)
        gene_expr = np.minimum(gene_expr, 15)

    mask = np.expand_dims(mask, axis=-1)
    raw_data = np.expand_dims(raw_data, axis=-1)
    gene_expr = np.expand_dims(gene_expr, axis=-1)
    sorted_pseudo_time = np.expand_dims(sorted_pseudo_time, axis=-1)
    masked_data = gene_expr * mask  
    filled_data = interp_multivar_data(masked_data, mask, interp=opt.init_fill)
    filled_data = filled_data * (1-mask) + masked_data
     
    return raw_data, filled_data, gene_expr, mask, sorted_pseudo_time, gene_name, cell_name