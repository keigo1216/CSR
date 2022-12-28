import numpy as np

class Coef_optim:
    """
    係数マップを最適化する

    Attributes
    ----------
    N : int
        入力画像の縦（または横）の画素数
    M : int
        作成する辞書の枚数
    B : int
        辞書の縦（または横）の画素数
    K : int
        学習枚数
    Rho : int
        ADMMの係数
    S : np.array
        size = (K, N, N)
        畳み込みスパースで表したい元画像
    X : np.array
        size = (K, M, N, N)
        係数マップ
    Y : np.array
        size = (K, M, N, N)
    U : np.array
        size = (K, M, N, N)
        双対変数
    D : np.array
        size = (M, N, N)
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
        self.N = args.N 
        self.M = args.M
        self.B = args.B
        self.K = args.K
        self.Rho = args.Rho
        self.Lam = args.Lam
        self.coef_loop = args.coef_loop

        self.S = S
        self.X = X
        # self.reset_parameters
        
    def reset_parameters(self):
        self.Y = self.X #制約条件のX-Y=0より
        self.U = np.zeros(shape=(self.K, self.M, self.N, self.N)) #双対変数Uの初期化


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
                self.update_Y(k)
                self.update_U(k)
            
        return self.X
        
    def update_X(self, k):
        """
        Xを更新する.
        更新の流れ次のようになっている.
        1. D, S, Y, Uを2DでFFT. 正規化が必ず必要.
        2. 逆行列の補助定理を使いながら最適なXに更新する.
        3. 逆フーリエ変換して最適なXを計算する

        Parameters
        ----------
        k : int
            画像kについて最適化する
        """

        #1.
        D_hat = np.fft.fft2(self.D,    norm="ortho") #(M, N, N)
        S_hat = np.fft.fft2(self.S[k], norm="ortho") #(N, N)
        Y_hat = np.fft.fft2(self.Y[k], norm="ortho") #(M, N, N)
        U_hat = np.fft.fft2(self.U[k], norm="ortho") #(M, N, N)

        #2
        D_hat_conjugation = np.conjugate(D_hat) #(M, N, N)
        left_side = D_hat_conjugation * S_hat + self.Rho * (Y_hat - U_hat) #(M, N, N)
        diag_inv = 1 / (1 + np.sum(D_hat * D_hat_conjugation, axis=0)/self.Rho) #(N, N)
        # diag = 1 + np.sum(D_hat * D_hat_conjugation, axis=0) / self.Rho
        # diag_inv = 1 / diag
        X_hat = left_side / self.Rho - D_hat_conjugation * (diag_inv * np.sum(D_hat * left_side, axis=0))/self.Rho**2 #(M, N, N)

        #3.
        X = np.fft.ifft2(X_hat, norm="ortho").real #計算機の誤差で若干虚数成分も残るが気にしない
        self.X[k] = X #当該箇所のXの値を更新する

    def update_Y(self, k):
        """
        Yを近接写像を用いて行絵rつの要素ごとに最適化する.

        Parameters
        ----------
        k : int
            画像kについて最適化する
        """
        V = self.X[k] + self.U[k]
        self.Y[k] = np.sign(V) * np.clip(np.abs(V)-self.Lam/self.Rho, 0, float("inf")) #ソフト閾値関数で最適化
    
    def update_U(self, k):
        """
        Uを更新する.

        Parameters
        ----------
        k : int
            画像kについて最適化する
        """
        self.U[k] = self.U[k] + self.X[k] - self.Y[k]