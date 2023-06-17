import logging
import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops

import tqdm
import numpy as np
import argparse
from omegaconf import OmegaConf
from copy import deepcopy
import torch
from torch import dropout, nn


from src.utils.misc import plot_causal_matrix, reproduc, get_dataset_information
from src.utils.logger import MyLogger

from src.data.prepare_data import prepare_real_data
from src.training import GRODE
from datetime import datetime
from src.utils.eval import pred_eval, infer_eval
import os
from einops import rearrange
import anndata
import scanpy as sc
import pandas as pd


def main(opt, dpf=True, glf=True, device="cuda"):
    reproduc(**opt.reproduc)
    timestamp = datetime.now().strftime("_%Y_%m%d_%H%M%S_%f")
    task_name = opt.task_name + timestamp
    proj_path = opj(opt.dir_name, task_name)
    log = MyLogger(log_dir=proj_path, **opt.log)
    log.log_opt(opt)

    raw_data, filled_data, original_data, mask, \
    time_list, gene_name_list, cell_name_list = prepare_real_data(opt.data, opt.grode.task)


    
    if hasattr(opt, "grode"):
        if opt.grode.task == 'non_celltype_GRN':
            grode = GRODE(opt.grode, log, device=device)
            network, pred_data = grode.train(opt, original_data, mask, original_data, raw_data, 
                                             gene_name_list = gene_name_list,
                                             cell_name_list = cell_name_list, glf = False, dpf = False)   
        elif opt.grode.task == 'celltype_GRN':
            grode = GRODE(opt.grode, log, device=device)
            network, pred_data = grode.train(opt, filled_data, mask, original_data, raw_data, 
                                             gene_name_list = gene_name_list,
                                             cell_name_list = cell_name_list, glf = False, dpf = False)   
        elif opt.grode.task == 'imputation':
            grode = GRODE(opt.grode, log, device=device)
            network, pred_data = grode.train(opt, filled_data, mask, original_data, raw_data, 
                                             gene_name_list = gene_name_list,
                                             cell_name_list = cell_name_list, glf = False, dpf = False)   
        else:
             raise Exception("The 'task' parameter make a mistake.")

        print("complete! ")   
        
        if glf:
            EPratio, EP, AUPRratio, AUPR, roc = infer_eval(network, opt.data.data_dir)
            
        if dpf:
            scores_kmeans = pred_eval(pred_data, raw_data, cell_name_list, opt.data.data_name, opt.reproduc.seed)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    parser = argparse.ArgumentParser(description='grode')
    parser.add_argument('-opt', type=str, default=opj(opd(__file__),
                        'demo.yaml'), help='yaml file path')
    parser.add_argument('-g', help='availabel gpu list.', default='1', type=str)
    parser.add_argument('-seed', help='set random seed.', default=None, type=int)
    parser.add_argument('-data', help='used data name. Select from (mESC, hESC, hHep, mDC, \
                        mHSC-E, mHSC-GM, mHSC-L, Embryos, Klein)', default=None, type=str)
    parser.add_argument('-n_nodes', help='the number of genes in the dataset', default=None, type=int)
    parser.add_argument('-k', help='lambda_k, which is the same as beta in the paper.', default=None, type=float)
    parser.add_argument('-s', help='lambda_s, which is the same as gamma in the paper.', default=None, type=float)
    parser.add_argument('-total_epoch', help='total epoch of the GRODE', default=None, type=int)
    parser.add_argument('-sp', help='supervision policy', default=None, type=str)
    parser.add_argument('-fp', help='fill policy', default=None, type=str)
    parser.add_argument('-cell_num', help='the number of cells used', default=None, type=int)
    parser.add_argument('-task', help='the task of grode', default=None, type=int)
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-log', action='store_true')
    args = parser.parse_args()
    opt = OmegaConf.load(args.opt)  
    
    if args.g == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        device = "mps"
    elif args.g == "cpu":
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.g
        device = "cuda"
    if args.seed != None:
        opt.reproduc.seed = args.seed
    if args.data != None and args.n_nodes != None:
        opt.data.data_name = args.data
        opt.data.data_dir = '/home/djy/GRODE/data/rdata' + args.data
        opt.grode.n_nodes = args.n_nodes
    if args.sp != None:
        opt.grode.supervision_policy = args.sp
    if args.fp != None:
        opt.grode.fill_policy = args.fp
    if args.k != None:
        opt.grode.data_pred.lambda_k_start = args.k
        opt.grode.data_pred.lambda_k_end = args.k
    if args.s != None:
        opt.grode.graph_discov.lambda_s_start = args.s
        opt.grode.graph_discov.lambda_s_end = args.s
    if args.cell_num != None:
        opt.data.cell_num = args.cell_num
    if args.task != None:
        opt.grode.task = args.task
        
    main(opt, glf=True, dpf=False, device=device)
    torch.cuda.empty_cache()
