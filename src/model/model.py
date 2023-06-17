from einops import rearrange
import torch
import torch.nn as nn
import math
import numpy as np
from torch.autograd import Variable
import torchdiffeq as ode

from src.model.ode import * 

class TransMLP(nn.Module):
    def __init__(self, in_dim, hid_dim, ebd_dim, out_dim, n_nodes=1, 
                 pos_ebd_flag=True, graph_learner_flag=True, pred_model='ode'):
        super(TransMLP, self).__init__()

        self.n_nodes = n_nodes
        self.pos_ebd_flag = pos_ebd_flag
        self.graph_learner_flag = graph_learner_flag
        self.pred_model = pred_model
        
        self.exp_ebd = torch.nn.Sequential(
            nn.Linear(1, ebd_dim),
            nn.Tanh(),
            nn.Linear(ebd_dim, ebd_dim),
            nn.Tanh(),
        )

        if pos_ebd_flag:
            self.pos_ebd = torch.nn.Sequential(
                nn.Linear(1, ebd_dim),
                nn.Tanh(),
                nn.Linear(ebd_dim, ebd_dim),
                nn.Tanh(),
            )
        
        self.fitting_model_in = torch.nn.Sequential(
                nn.Linear(in_dim * ebd_dim, hid_dim),
                nn.Tanh(),
                nn.Linear(hid_dim, ebd_dim),
                nn.Tanh())
        
        self.fc_mu = nn.Linear(ebd_dim, ebd_dim)
        self.fc_std = nn.Linear(ebd_dim, ebd_dim)

        if pred_model == 'ode':
            self.ode_func = ODEFunc(ebd_dim, hid_dim) 
            self.neural_dynamic_layer = ODEBlock(odefunc=self.ode_func)  
            self.output_layer = nn.Linear(ebd_dim, 1)
        elif pred_model == 'mlp':
            self.fitting_model_out = torch.nn.Sequential(
                nn.Linear(ebd_dim, hid_dim),
                nn.Tanh(),
                nn.Linear(hid_dim, out_dim + 1))
            
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def _one_minus_A_t(self, adj):
        adj_normalized = torch.Tensor(np.eye(adj.shape[0])).to(adj) - (adj.transpose(0, 1))
        return adj_normalized

    def forward(self, x, x_t, y_t, adj):
        b, t, n, d = x.shape
        x = self.exp_ebd(x)
        x_ = rearrange(x, "b t n d -> b n (t d)")

        if self.pos_ebd_flag:
            x_t_ = self.pos_ebd(x_t)
            x_t_ = x_t_[:, None].expand(-1, n, -1, -1)
            x_t_ = rearrange(x_t_, "b n t d -> b n (t d)")

        if self.pos_ebd_flag:
            y_hidden = self.fitting_model_in(x_ + x_t_)
        else:
            y_hidden = self.fitting_model_in(x_)
            
        mu = self.fc_mu(y_hidden)
        log_var = self.fc_std(y_hidden)
        
        mask = Variable(torch.from_numpy(np.ones(self.n_nodes) - np.eye(self.n_nodes)).float(), requires_grad=False).to(x)
        init_graph = adj * mask
        
        if self.graph_learner_flag:
            adj_A_t = self._one_minus_A_t(init_graph)
            adj_A_t_ = adj_A_t[None].expand(b, -1, -1)
            mu = torch.matmul(adj_A_t_, mu)
            log_var = torch.matmul(adj_A_t_, log_var)
        
        var = torch.exp(log_var)
        sigma = torch.sqrt(var + 1e-10)
        z = torch.randn(size = mu.size())
        z = z.type_as(mu) 
        z = mu + sigma*z
        
        if self.graph_learner_flag:
            adj_A_t_inv = torch.inverse(adj_A_t)[None].expand(b, -1, -1)
            z_inv = torch.matmul(adj_A_t_inv, z)
            z = z_inv
            
        if not self.graph_learner_flag:
            z = torch.matmul(adj * mask, z)

        if self.pred_model == 'ode':
            t = torch.linspace(0., 1, y_t.shape[1] + 1).to(y_t)
            y_pred = self.neural_dynamic_layer(z, t)
            y_pred = self.output_layer(y_pred)
            y_pred = rearrange(y_pred, "t b n d -> b t n d")
        elif self.pred_model == 'mlp':
            y_pred = self.fitting_model_out(z)
            y_pred = rearrange(y_pred, "b n (t d) -> b t n d", d=d)
        else:
            raise Exception(
                f"Unknown pred model, please choose 'ode' or 'mlp'.")

        return y_pred, mu, log_var
