import torch
from opt import args
import torch.nn as nn
import torch.nn.functional as F
from opt import args

class sample_aware_network(nn.Module):
    def __init__(self, input_dim, hidden_dim, act, n_num):
        super(sample_aware_network, self).__init__()
        self.AEa = nn.Linear(input_dim, hidden_dim)
        self.AEb = nn.Linear(input_dim, hidden_dim)

        self.SEa = nn.Linear(n_num, hidden_dim)
        self.SEb = nn.Linear(n_num, hidden_dim)

        self.alpha = nn.Parameter(torch.Tensor(1, ))
        self.alpha.data = torch.tensor(0.99999).to(args.device)

        self.pos_weight = torch.ones(n_num * 2).to(args.device)
        self.pos_neg_weight = torch.ones([n_num * 2, n_num * 2]).to(args.device)

        if act == "ident":
            self.activate = lambda x: x
        if act == "sigmoid":
            self.activate = nn.Sigmoid()

    def forward(self, x, A):
        Za = self.activate(self.AEa(x))
        Zb = self.activate(self.AEb(x))

        Za = F.normalize(Za, dim=1, p=2)
        Zb = F.normalize(Zb, dim=1, p=2)

        Ea = F.normalize(self.SEa(A), dim=1, p=2)
        Eb = F.normalize(self.SEb(A), dim=1, p=2)

        return Za, Zb, Ea, Eb
