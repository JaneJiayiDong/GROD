
import tqdm
import numpy as np
from copy import deepcopy
import torch
from torch import dropout, nn

from src.model.model import TransMLP
from src.utils.misc import plot_causal_matrix_in_training, log_time_series, extractEdgesFromMatrix, get_dataset_information
from src.utils.logger import MyLogger
from src.utils.eval import infer_eval, pred_eval

from src.data.generate_slice import generate_slice
from src.data.data_loader import load_dataset, split_dataset
from datetime import datetime
from torchinfo import summary

import os
from einops import rearrange
import anndata
import scanpy as sc
from sklearn.cluster import KMeans
from src.inferring import infer_trajectory

class GRODE(object):
    def __init__(self, args, log, device="cuda"):
        self.log: MyLogger = log
        self.args = args
        self.device = device

        # model
        self.fitting_model = TransMLP(self.args.input_step * self.args.data_dim,
                                        self.args.data_pred.mlp_hid,
                                        self.args.data_pred.ebd_dim,
                                        self.args.data_dim * self.args.data_pred.pred_step,
                                        self.args.n_nodes,
                                        self.args.pos_ebd_flag, 
                                        self.args.graph_learner_flag, 
                                        self.args.pred_model).to(self.device)

        # pred data
        self.data_pred_loss = nn.MSELoss(reduce=False)
        self.data_pred_optimizer = torch.optim.Adam(self.fitting_model.parameters(),
                                                    lr=self.args.data_pred.lr_data_start,
                                                    weight_decay=self.args.data_pred.weight_decay)
        if "every" in self.args.fill_policy:
            lr_schedule_length = int(self.args.fill_policy.split("_")[-1])
        else:
            lr_schedule_length = self.args.total_epoch
        gamma = (self.args.data_pred.lr_data_end / self.args.data_pred.lr_data_start) ** (1 / lr_schedule_length)
        self.data_pred_scheduler = torch.optim.lr_scheduler.StepLR(
            self.data_pred_optimizer, step_size=1, gamma=gamma)
        
        # graph learner
        self.graph = nn.Parameter(((torch.randn([self.args.n_nodes, self.args.n_nodes]) * 0)).to(self.device))
        self.graph_optimizer = torch.optim.Adam([{'params':self.graph}], 
                                                 lr=self.args.graph_discov.lr_graph_start)
        gamma = (self.args.graph_discov.lr_graph_end / self.args.graph_discov.lr_graph_start) ** (1 / self.args.total_epoch)
        self.graph_scheduler = torch.optim.lr_scheduler.StepLR(self.graph_optimizer, step_size=1, gamma=gamma)
        
        # learning rate 
        end_lmds, start_lmds = self.args.graph_discov.lambda_s_end, self.args.graph_discov.lambda_s_start
        self.lambda_s_gamma = (end_lmds / start_lmds) ** (1 / self.args.total_epoch)
        self.lambda_s = start_lmds
        end_lmdk, start_lmdk = self.args.data_pred.lambda_k_end, self.args.data_pred.lambda_k_start
        self.lambda_k_gamma = (end_lmdk / start_lmdk) ** (1 / self.args.total_epoch)
        self.lambda_k = start_lmdk

    def latent_data_pred(self, x, y, x_t, y_t, x_mask, y_mask):  
        bs, t, n, d = x.shape    
        self.fitting_model.train()
        self.data_pred_optimizer.zero_grad()

        y_pred, mu, log_var = self.fitting_model(x * x_mask, x_t, y_t, self.graph)
        
        recon_loss = (self.data_pred_loss(y * y_mask, y_pred[:, 1:] * y_mask)).mean() / torch.mean(y_mask)
        kl_loss = self.lambda_k * torch.mean(torch.mean(0.5 * torch.sum(mu ** 2 + torch.exp(log_var) - log_var - 1, -1), -1), -1) 

        loss =  recon_loss +  kl_loss
        loss.backward()
        self.data_pred_optimizer.step()
        return y_pred, loss, recon_loss, kl_loss

    def graph_discov(self, x, y, x_t, y_t, x_mask, y_mask):
        bs, t, n, d = x.shape
        self.graph_optimizer.zero_grad()
        y_pred, mu, log_var = self.fitting_model(x * x_mask, x_t, y_t, self.graph) # x_mask

        sparsity_loss = self.lambda_s * torch.mean(torch.abs(self.graph))
        recon_loss = (self.data_pred_loss(y * y_mask, y_pred[:, 1:] * y_mask)).mean() / torch.mean(y_mask)
    
        kl_loss = self.lambda_k * torch.mean(torch.mean(0.5 * torch.sum(mu ** 2 + torch.exp(log_var) - log_var - 1, -1), -1), -1)  
        loss = recon_loss +  kl_loss + sparsity_loss 
        
        loss.backward()
        self.graph_optimizer.step()
        return loss, sparsity_loss, recon_loss, kl_loss

    def train(self, opt, filled_data, observ_mask, original_data, raw_data, cell_name_list, time_list=None, gene_name_list=None,
              true_cm=None, viz=True, glf=False, dpf=False, infer_trajectories=False):

        original_data = torch.from_numpy(original_data).float().to(self.device)
        data = torch.from_numpy(filled_data).float().to(self.device)
        observ_mask  = original_mask = torch.from_numpy(observ_mask).float().to(self.device)
        data_pred = deepcopy(data)
        
        if time_list is None:
            time_list = torch.linspace(0., 1, original_data.shape[0]).unsqueeze(-1).to(self.device)
        else:
            time_list = torch.from_numpy(time_list).float().to(self.device)

        if self.args.supervision_policy == "masked":
            print("Using masked supervision for data prediction...")
        elif self.args.supervision_policy == "full":
            print("Using full supervision for data prediction......")
            observ_mask = torch.ones_like(observ_mask)
        elif "masked_before" in self.args.supervision_policy:
            print(f"Using masked supervision for data prediction ({self.args.supervision_policy:s})......")

        data_pred_step = 0
        graph_discov_step = 0

        if viz:
            pbar = tqdm.tqdm(total=self.args.total_epoch)
        
        # Data prediction
        for epoch_i in range(self.args.total_epoch):
            
            # data updating process
            if "rate" in self.args.fill_policy:
                update_rate = float(self.args.fill_policy.split("_")[1])
                update_after = int(self.args.fill_policy.split("_")[3])
                if epoch_i+1 > update_after:
                    if epoch_i == update_after:
                        print("Data update started!", epoch_i)
                    data = data * (1 - update_rate) + data_pred * update_rate
                    
            # whether mask data or not
            if "masked_before" in self.args.supervision_policy:
                masked_before = int(self.args.supervision_policy.split("_")[2])
                if epoch_i == masked_before:
                    print("Using full supervision for data prediction......")
                    observ_mask = torch.ones_like(original_mask)
            
            # data preprocess
            dataset = generate_slice(data, observ_mask, time_list,
                            input_step=self.args.input_step, pred_step=self.args.data_pred.pred_step)
            train_data, val_data, test_data = split_dataset(dataset)
            self._data = load_dataset(train_data, val_data, test_data, batch_size=self.args.batch_size)

            if hasattr(self.args, "data_pred"):
                data_pred = deepcopy(data) # masked data points are predicted
                data_pred_all = deepcopy(data)
                train_iterator = self._data['train_loader'].get_iterator()  

                for batch_idx,  (x, y, x_t, y_t, x_mask, y_mask, t_id) in enumerate(train_iterator):
                    data_pred_step += self.args.batch_size
                    y_pred, loss, recon_loss, kl_loss = self.latent_data_pred(x, y, x_t, y_t, x_mask, y_mask)
                    data_pred[t_id] = (y_pred[:, 0]*(1-x_mask[:, -1]) + x[:, -1]*x_mask[:, -1]).clone().detach() 
                    data_pred_all[t_id] = y_pred[:, 0].clone().detach()             
                    
                    self.log.log_metrics({"latent_data_pred/pred_loss": loss.item()}, data_pred_step)
                    if viz:
                        pbar.set_postfix_str(f"S1: recon loss={recon_loss.item():.4f}, kl_loss={kl_loss.item():.4f}, spr=******")

                current_data_pred_lr = self.graph_optimizer.param_groups[0]['lr']
                self.log.log_metrics({"data_pred/lr": current_data_pred_lr}, data_pred_step)
                self.data_pred_scheduler.step()

                mse_pred_to_original = self.data_pred_loss(original_data, data_pred).mean()
                self.log.log_metrics({"latent_data_pred/mse_pred_to_original": mse_pred_to_original}, data_pred_step)
            
            # Graph Discovery
            if hasattr(self.args, "graph_discov"):
                train_iterator = self._data['train_loader'].get_iterator()  
                for batch_idx,  (x, y, x_t, y_t, x_mask, y_mask, t_id) in enumerate(train_iterator):
                    graph_discov_step += self.args.batch_size
                    if hasattr(self.args, "disable_graph") and self.args.disable_graph:
                        pass
                    else:
                        loss, sparsity_loss, recon_loss, kl_loss = self.graph_discov(x, y, x_t, y_t, x_mask, y_mask)
                        self.log.log_metrics({"graph_discov/sparsity_loss": sparsity_loss.item(),
                                            "graph_discov/data_loss": recon_loss.item(),
                                            "graph_discov/kl_loss": kl_loss.item(),
                                            "graph_discov/total_loss": loss.item()}, graph_discov_step)
                        if viz:
                            pbar.set_postfix_str(f"S2: recon loss={recon_loss.item():.4f}, kl loss={kl_loss.item():.4f}, spr={sparsity_loss.item():.4f}")
                    
                self.graph_scheduler.step()
                current_graph_disconv_lr = self.graph_optimizer.param_groups[0]['lr']
                self.log.log_metrics({"graph_discov/lr": current_graph_disconv_lr}, graph_discov_step)
                    
            if viz:
                pbar.update(1)
            self.lambda_k *= self.lambda_k_gamma
            self.lambda_s *= self.lambda_s_gamma

            calc, val = self.args.causal_thres.split("_")
            if calc == "value":
                threshold = float(val)
            else:
                raise NotImplementedError
            
            mat = self.graph.detach().cpu().numpy()
            if (epoch_i+1) % self.args.show_graph_every == 0:
                if glf:
                    if gene_name_list == None:
                        gene_name_list = np.arange(0, mat.shape[0], 1)
                    network = extractEdgesFromMatrix(mat, gene_name_list, TFmask=None)        
                    EPratio, EP, AUPRratio, AUPR, roc = infer_eval(network, opt.data.data_dir)
                    print(np.round(EPratio,4))
                    print(np.round(EP,4))
                    print(np.round(AUPRratio,4))
                    print(np.round(AUPR,4))
                    print(np.round(roc,4))
                
                if dpf:
                    adata_values = data.squeeze(-1).detach().cpu().numpy()
                    scores_kmeans = pred_eval(adata_values, raw_data, cell_name_list, opt.data.data_name, opt.reproduc.seed)
                    print("KMEANS:", scores_kmeans['accuracy'], scores_kmeans['f1_score'], 
                          scores_kmeans['adj_rand'], scores_kmeans['nmi'])
        
        
        if infer_trajectories:
            train_iterator = self._data['train_loader'].get_iterator()  
            pred_data = data.squeeze(-1).detach().cpu().numpy()
            infer_trajectory(train_iterator, self.fitting_model, self.graph, raw_data, cell_name_list, opt.data.data_name, pred_data)
            
        
        mat = self.graph.detach().cpu().numpy()
        if gene_name_list == None:
            gene_name_list = np.arange(0, mat.shape[0], 1)
        network = extractEdgesFromMatrix(mat, gene_name_list, TFmask=None)        
        
        return network, data.squeeze(-1).detach().cpu().numpy()