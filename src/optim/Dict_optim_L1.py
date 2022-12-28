import numpy as np
from .Dict_optim import Dict_optim

class Dict_optim_L1(Dict_optim):
    def __init__(self, D, S, args):
        super().__init__(D, S, args)
        self.D_X_conv = np.zeros([self.N, self.N])
    
    def reset_parameters(self):
        X_hat = np.fft.fft2(np.sum(self.X, axis=0), norm="ortho") #(M, N, N)
        D_hat = np.fft.fft2(self.D, norm="ortho") #(M, N, N)
        D_X_conv_hat = np.sum(D_hat * X_hat, axis=0) #(N, N)
        self.D_X_conv = np.fft.ifft2(D_X_conv_hat, norm="ortho").real #(N, N)

        self.G_0 = self.D_X_conv - np.sum(self.S, axis=0) #(N, N)
        self.G_1 = self.D #(M, N, N)
        self.U_0 = np.zeros_like(self.G_0) #(N, N)
        self.U_1 = np.zeros_like(self.G_1) #(M, N, N)

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
            self.update_D()
            self.update_G_0()
            self.update_G_1()
            self.update_U_0()
            self.update_U_1()
        return self.D
    
    def update_D(self):
        """
        L1誤差で評価した目的関数の辞書を更新する
        更新の流れは次のようになっている.
        1. 複数の画像を一枚の画像にする. 本来はしてはいけないけどめんどくさいから代替案として採用. 極端に精度が落ちることはなさそうな気がする.
        1. X, S, G, Uを2DでFFT. 正規化が必ず必要.
        2. 複数枚の画像を一枚の画像に直す. 係数の最適化のように上手いことできないから代替案（原著の(3)のやり方）
        3. 逆行列の補助定理を使いながら最適なD_hatを計算する.
        4. 逆フーリエ変換して最適なDを計算する
        """

        #1.
        tiled_X = np.sum(self.X, axis=0) #(M, N, N)
        tiled_S = np.sum(self.S, axis=0) #(N, N)

        #2.
        X_hat = np.fft.fft2(tiled_X, norm="ortho") #(M, N, N)
        S_hat = np.fft.fft2(tiled_S, norm="ortho") #(N, N)
        G_0_hat = np.fft.fft2(self.G_0, norm="ortho") #(N, N)
        G_1_hat = np.fft.fft2(self.G_1, norm="ortho") #(M, N, N)
        U_0_hat = np.fft.fft2(self.U_0, norm="ortho") #(N, N)
        U_1_hat = np.fft.fft2(self.U_1, norm="ortho") #(M, N, N)

        #3.
        X_hat_conjugation = np.conjugate(X_hat) #(M, N, N)
        left_side = X_hat_conjugation * (G_0_hat + S_hat - U_0_hat) + G_1_hat - U_1_hat
        diag_inv = 1 / (1 + np.sum(X_hat * X_hat_conjugation, axis=0)) #(N, N)
        D_hat = left_side/self.Rho - X_hat_conjugation * (diag_inv * np.sum(X_hat * left_side, axis=0))

        #4.
        self.D = np.fft.ifft2(D_hat, norm="ortho").real

    def update_G_0(self):
        """
        G_0を近接写像を用いて更新する
        """

        X_hat = np.fft.fft2(np.sum(self.X, axis=0), norm="ortho") #(M, N, N)
        D_hat = np.fft.fft2(self.D, norm="ortho") #(M, N, N)
        D_X_conv_hat = np.sum(D_hat * X_hat, axis=0) #(N, N)
        self.D_X_conv = np.fft.ifft2(D_X_conv_hat, norm="ortho").real #(N, N)

        V = self.D_X_conv - np.sum(self.S, axis=0) + self.U_0
        self.G_0 = np.sign(V) * np.clip(np.abs(V)-self.Lam/self.Rho, 0, float("inf")) #ソフト閾値関数で最適化
    
    def update_G_1(self):
        """
        G_1を近接写像を用いて更新する
        """

        Y = self.D + self.U_1

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
    
    def update_U_0(self):
        """
        双対変数U_0を更新する
        """
        self.U_0 = self.U_0 + self.D_X_conv - self.G_0 - np.sum(self.S, axis=0)
    
    def update_U_1(self):
        """
        双対変数U_1を更新する
        """
        self.U_1 = self.U_1 + self.D - self.G_1
