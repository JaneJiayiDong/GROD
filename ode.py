
import torch
import torch.nn as nn
import torchdiffeq as ode


class ODEFunc(nn.Module):  # A kind of ODECell in the view of RNN
    def __init__(self, ebd_dim, hid_dim, dropout=0):
        super(ODEFunc, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.fc = torch.nn.Sequential(
                nn.Linear(ebd_dim, hid_dim),
                nn.Tanh(),
                nn.Linear(hid_dim, ebd_dim),
                nn.Tanh())

    def forward(self, t, x):
        # x: num_node, hidden_size
        # self.graph: num_node, hidden_size
        x = self.fc(x)
        return x

class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol=.01, atol=.001, method='dopri5', adjoint=False, terminal=False): 
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.adjoint = adjoint
        self.terminal = terminal

    def forward(self,  x, vt):
        integration_time_vector = vt.type_as(x)

        if self.adjoint:
            out = ode.odeint_adjoint(self.odefunc, x, integration_time_vector,
                                     rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = ode.odeint(self.odefunc, x, integration_time_vector,
                             rtol=self.rtol, atol=self.atol, method=self.method)

        return out[-1] if self.terminal else out  # 100 * 400 * 10


