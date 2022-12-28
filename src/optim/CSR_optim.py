import numpy as np
from .Coef_optim import Coef_optim
from .Coef_optim_L1 import Coef_optim_L1
from .Dict_optim import Dict_optim
from .Dict_optim_L1 import Dict_optim_L1
from .Dict_optim_consensus_L2 import Dict_optim_consensus_L2
from .Dict_optim_consensus_L1 import Dict_optim_consensus_L1
import matplotlib.pyplot as plt

class CSR_optim:
    def __init__(self, D, X, S, args):
        self.D = D
        self.X = X
        self.S = S
        self.B = args.B
        self.args = args

        # self.coef_optim = Coef_optim(X, S, args)
        self.coef_optim = Coef_optim_L1(X, S, args)
        if args.is_consensus is True:
            self.dict_optim = Dict_optim_consensus_L2(D, S, args)
        else:
            self.dict_optim = Dict_optim(D, S, args)
        # self.dict_optim = Dict_optim_L1(D, S, args)
        self.dict_optim = Dict_optim_consensus_L1(D, S, args)

    def optimization(self):
        """
        D, Xの最適化を1epoch行う
        """
        self.X = self.coef_optim.coef_update(self.D, self.args.coef_loop)
        self.D = self.dict_optim.dict_update(self.X, self.args.dict_loop)

        return self.D, self.X