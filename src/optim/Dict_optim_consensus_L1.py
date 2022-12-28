import numpy as np
from .Dict_optim import Dict_optim

class Dict_optim_consensus_L1(Dict_optim):
    def __init__(self, D, S, args):
        super().__init__(D, S, args)
        self.D_X_conv = np.zeros([self.K, self.N, self.N])
        self.D_k = np.zeros([self.K, self.M, self.N, self.N])
    
    def reset_parameters(self):
        D_hat = np.fft.fft2(self.D, norm="ortho") #(M, N, N)
        X_hat = np.fft.fft2(self.X, norm="ortho") #(K, M, N, N)
        D_X_conv_hat = np.sum(D_hat * X_hat, axis=1) #(K, N, N)
        self.D_X_conv = np.fft.ifft2(D_X_conv_hat, norm="ortho").real #(K, N, N)

        self.G_0 = self.D_X_conv - self.S #G_0を\sum_m d_m * s_m - s_kで初期化, (K, N, N)
        self.G_1 = self.D #G_1をDで初期化する #(M, N, N)
        self.H_0 = np.zeros_like(self.G_0) #(K, N, N)
        self.H_1 = np.zeros_like(self.D_k) #(K, M, N, N)
    
    def dict_update(self, X, iteration):
        """
        Parameters
        ----------
        X : np.array
            size = (K, M, N, N)
            係数マップ
        iteration : int
            辞書の最適化を行う回数

        Returns
        -------
        D : np.array
            size = (M, N, N)
            最適化した辞書を返す
        """
        
        self.X = X
        self.reset_parameters()

        for i in range(iteration):
            for k in range(self.K):
                self.update_D(k)
            self.update_G_0()
            self.update_G_1()
            self.update_H_0()
            self.update_H_1()
        
        self.D = self.D_k[0]
        return self.D
    
    def update_D(self, k):
        """
        画像kの辞書をコンセンサス方式で更新する
        更新の流れは次のようになっている.
        1. X, S, G_0, G_1, H_0, H_1を2DでFFT. 正規化が必ず必要.
        2. 逆行列の補助定理を使いながら最適なD_hatを計算する.
        3. 逆行列の補助定理を使いながら最適なD_hatを計算する.
        4. 逆フーリエ変換して最適なDを計算する
        """

        #1. 
        X_hat = np.fft.fft2(self.X[k], norm="ortho") #(M, N, N)
        S_hat = np.fft.fft2(self.S[k], norm="ortho") #(N, N)
        G_0_hat = np.fft.fft2(self.G_0[k], norm="ortho") #(N, N)
        G_1_hat = np.fft.fft2(self.G_1, norm="ortho") #(M, N, N)
        H_0_hat = np.fft.fft2(self.H_0[k], norm="ortho") #(N, N)
        H_1_hat = np.fft.fft2(self.H_1[k], norm="ortho") #(M, N, N)

        #2.
        X_hat_conjugation = np.conjugate(X_hat) #(M, N, N)
        left_side = X_hat_conjugation*(G_0_hat+S_hat-H_0_hat) + G_1_hat-H_1_hat #(M, N, N)
        diag_inv = 1 / (1 + np.sum(X_hat * X_hat_conjugation, axis=0)) #(N, N)
        D_hat = left_side - X_hat_conjugation * (diag_inv * np.sum(X_hat * left_side, axis=0))

        #4.
        self.D_k[k] = np.fft.ifft2(D_hat, norm="ortho").real

    def update_G_0(self):
        """
        G_0を近接写像で更新する
        """
        D_hat = np.fft.fft2(self.D_k, norm="ortho") #(K, M, N, N)
        X_hat = np.fft.fft2(self.X, norm="ortho") #(K, M, N, N)
        D_X_conv_hat = np.sum(D_hat * X_hat, axis=1) #(K, N, N)
        self.D_X_conv = np.fft.ifft2(D_X_conv_hat, norm="ortho").real #(K, N, N)

        V = self.D_X_conv - self.S + self.H_0
        self.G_0 = np.sign(V) * np.clip(np.abs(V)-1/self.Rho, 0, float("inf")) #ソフト閾値関数で最適化

    def update_G_1(self):
        """
        G_1を近接写像で更新する
        """
        Y = np.sum(self.D_k + self.H_1, axis=0) / self.K
        
        #PP^{T}は0パディングすればOK
        Y = np.pad(Y[:, :self.B, :self.B], ((0, 0), (0, self.N-self.B), (0, self.N-self.B)))

        #ノルムの正規化
        #もっと綺麗な書き方あるはずだけど妥協しました
        norm_Y = np.sum(np.sqrt(Y**2), axis=(1, 2)) #各辞書（m）のl2ノルムを取得
        for m, norm_Y_m in enumerate(norm_Y):
            if norm_Y_m < 1: #1よりノルムが小さい辞書は無視する
                continue
            Y[m] /= norm_Y_m
        self.G_1 = Y
    
    def update_H_0(self):
        """
        H_0を更新する
        """
        self.H_0 = self.H_0 + self.D_X_conv - self.G_0 - self.S
    
    def update_H_1(self):
        """
        H_1を更新する
        """
        self.H_1 = self.H_1 + self.D_k - self.G_1