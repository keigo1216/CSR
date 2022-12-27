import numpy as np
from .Dict_optim import Dict_optim

class Dict_optim_consensus(Dict_optim):
    def __init__(self, D, S, args):
        super().__init__(D, S, args)
        self.D_k = np.zeros(shape=(self.K, self.M, self.N, self.N)) #コンセンサス方式で使う変数
    
    def reset_parameters(self):
        self.G = self.D
        self.U = np.zeros(shape=(self.K, self.M, self.N, self.N))

    def dict_update(self, X):
        """
        Parameters
        ----------

        X : np.array
            size = (K, M, N, N)
            係数マップ
        
        Returns
        ------
        D : np.array
            size = (M, N, N)
            最適化した辞書を返す
        """

        self.X = X

        #画像ごとに辞書を更新する
        for k in range(self.K):
            self.update_D(k)

        #Gを更新（Kには依存していないことに注意）
        self.update_G()

        #双対変数を画像ごとに最適化する 
        # for k in range(self.K):
        self.update_U(k)
        
        self.D = self.D_k[0]
        return self.D

    def update_D(self, k):
        """
        画像kについて辞書を最適化する
        """

        X_hat = np.fft.fft2(self.X[k], norm="ortho")
        S_hat = np.fft.fft2(self.S[k], norm="ortho")
        G_hat = np.fft.fft2(self.G,    norm="ortho")
        U_hat = np.fft.fft2(self.U[k], norm="ortho")

        X_hat_conjugation = np.conjugate(X_hat) #(M, N, N)
        left_side = X_hat_conjugation*S_hat + self.Rho * (G_hat - U_hat) #(M, N, N)
        diag_inv = 1 / (1 + np.sum(X_hat * X_hat_conjugation, axis=0)/self.Rho) #(N, N)
        D_hat = left_side/self.Rho - X_hat_conjugation * (diag_inv * np.sum(X_hat * left_side, axis=0))/self.Rho

        self.D_k[k] = np.fft.ifft2(D_hat, norm="ortho").real
    
    def update_G(self):
        """
        Gを近接写像で更新する
        """
        Y = np.sum(self.D_k + self.U, axis=0) / self.K

        #PP^{T}は0パディングすればOK
        Y = np.pad(Y[:, :self.B, :self.B], ((0, 0), (0, self.N-self.B), (0, self.N-self.B)))

        #ノルムの正規化
        #もっと綺麗な書き方あるはずだけど妥協しました
        norm_Y = np.sum(np.sqrt(Y**2), axis=(1, 2)) #各辞書（m）のl2ノルムを取得
        for m, norm_Y_m in enumerate(norm_Y):
            if norm_Y_m < 1: #1よりノルムが小さい辞書は無視する
                continue
            Y[m] /= norm_Y_m
        self.G = Y
    
    def update_U(self, k):
        """
        k番目の画像の双対変数を更新する
        """

        # self.U[k] = self.U[k] + self.D_k[k] - self.G
        self.U = self.U + self.D - self.G #(K, M, N, N)