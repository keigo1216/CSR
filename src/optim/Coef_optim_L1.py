import numpy as np
from .Coef_optim import Coef_optim

class Coef_optim_L1(Coef_optim):
    """
    L1誤差で評価する時の係数マップの最適化を行うクラス
    """
    def __init__(self, X, S, args):
        """
        Parameters
        ----------
        X : np.array
            size = (K, M, N, N)
            係数マップの初期値
        S : np.array
            size = (K, N, N)
            畳み込みスパース表現で表した原画像
        args : 定数が入っているインスタンス
        """
        super().__init__(X, S, args)
        self.D_X_conv = np.zeros([self.K, self.N, self.N]) #\sum_m d_m*x_mの計算結果を格納しておく
    
    def reset_parameters(self):
        """
        ADMMを行う前のパラメータの初期化

        Y_0 = DX - S
        Y_1 = S
        U_0, U_1は全ての要素が0の行列
        """
        D_hat = np.fft.fft2(self.D, norm="ortho") #(M, N, N)
        X_hat = np.fft.fft2(self.X, norm="ortho") #(K, M, N, N)
        S_hat = np.fft.fft2(self.S, norm="ortho") #(K, N, N)
        Y_0_hat = np.sum(D_hat * X_hat, axis=1) - S_hat #(K, N, N)

        self.Y_0 = np.fft.ifft2(Y_0_hat, norm="ortho").real #(K, N, N)
        self.Y_1 = self.X #(K, M, N, N)

        self.U_0 = np.zeros_like(self.Y_0) #(K, N, N)
        self.U_1 = np.zeros_like(self.Y_1) #(K, M, N, N)
    
    def coef_update(self, D, iteration):
        """
        Parameters
        ----------
        D : np.array
            size = (M, N, N)
            辞書（係数の最適化をする際は定数として扱う）
        iteration : int
            係数マップの最適化回数

        Returns
        -------
        X : np.array
            size = (K, M, N, N)
            最適化した係数マップを返す
        """

        self.D = D
        self.reset_parameters()

        for i in range(iteration):
            #最適化を一回回す
            #画像ごとに最適化する
            for k in range(self.K):
                self.update_X(k)
                self.update_Y_0(k)
                self.update_Y_1(k)
            self.update_U_0()
            self.update_U_1()
        
        return self.X
        
    def update_X(self, k):
        """
        Xを更新する
        更新の流れは次のようになっている
        1. D, S, Y, Uを2DでFFT. 正規化が必ず必要.
        2. 逆行列の補助定理を使いながら最適なXに更新する.
        3. 逆フーリエ変換して最適なXを計算する.

        Parameters
        ----------
        k : int
            画像kについて最適化する.
        """

        #1.
        D_hat = np.fft.fft2(self.D, norm="ortho") #(M, N, N)
        S_hat = np.fft.fft2(self.S[k], norm="ortho") #(N, N)
        Y_0_hat = np.fft.fft2(self.Y_0[k], norm="ortho") #(N, N)
        Y_1_hat = np.fft.fft2(self.Y_1[k], norm="ortho") #(M, N, N)
        U_0_hat = np.fft.fft2(self.U_0[k], norm="ortho") #(N, N)
        U_1_hat = np.fft.fft2(self.U_1[k], norm="ortho") #(M, N, N)

        #2.
        D_hat_conjugation = np.conjugate(D_hat) #(M, N, N)
        left_side = D_hat * (Y_0_hat + S_hat - U_0_hat) + (Y_1_hat - U_1_hat) #(M, N, N)
        diag_inv = 1 / (1 + np.sum(D_hat * D_hat_conjugation, axis=0)) #(N, N)
        X_hat = left_side - D_hat_conjugation * (diag_inv * np.sum(D_hat * left_side, axis=0)) #(M, N, N)

        #3.
        X = np.fft.ifft2(X_hat, norm="ortho").real
        self.X[k] = X
    
    def update_Y_0(self, k):
        """
        Y_0を近接写像を用いて行列の要素ごとに最適化する.

        Parameters
        ----------
        k : int
            画像kについて最適化する
        """
        D_hat = np.fft.fft2(self.D, norm="ortho") #(M, N, N)
        X_hat = np.fft.fft2(self.X[k], norm="ortho") #(M, N, N)
        D_X_conv_hat = np.sum(D_hat * X_hat, axis=0) #(N, N)
        self.D_X_conv[k] = np.fft.ifft2(D_X_conv_hat, norm="ortho").real #(N, N)

        V = self.D_X_conv[k] - self.S[k] + self.U_0[k] #(N, N)
        self.Y_0[k] = np.sign(V) * np.clip(np.abs(V)-1/self.Rho, 0, float("inf"))
    
    def update_Y_1(self, k):
        """
        Y_1を近接写像を用いて行列の要素ごとに最適化する

        Parameters
        ----------
        k : int
            画像kについて最適化する.
        """

        V = self.X[k] + self.U_1[k]
        self.Y_1[k] = np.sign(V) * np.clip(np.abs(V)-self.Lam/self.Rho, 0, float("inf"))
    
    def update_U_0(self):
        """
        U_0を更新する
        """

        self.U_0 = self.U_0 + self.D_X_conv - self.Y_0 - self.S

    def update_U_1(self):
        """
        U_1を更新する
        """

        self.U_1 = self.U_1 + self.X - self.Y_1