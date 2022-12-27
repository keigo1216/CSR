import numpy as np
from .Coef_optim import Coef_optim
from .Dict_optim import Dict_optim
from .Dict_optim_consensus import Dict_optim_consensus
import matplotlib.pyplot as plt

class CSR_optim:
    def __init__(self, D, X, S, args):
        self.D = D
        self.X = X
        self.S = S
        self.B = args.B
        self.args = args

        self.coef_optim = Coef_optim(X, S, args)
        if args.is_consensus is True:
            self.dict_optim = Dict_optim_consensus(D, S, args)
        else:
            self.dict_optim = Dict_optim(D, S, args)
        self.coef_optim.reset_parameters()
        self.dict_optim.reset_parameters()

    def optimization(self):
        """
        D, Xの最適化を1epoch行う
        """
        self.coef_optim.reset_parameters()
        self.dict_optim.reset_parameters()
        for i in range(self.args.coef_loop):
            self.X = self.coef_optim.coef_update(self.D)
        
        for i in range(self.args.dict_loop):
            self.D = self.dict_optim.dict_update(self.X)

        return self.D, self.X