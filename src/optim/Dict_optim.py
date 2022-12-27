import numpy as np

class Dict_optim:
    """
    辞書を最適化する

    Attrubutes
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
    D : np.array
        size = (M, N, N)
    G : np.array
        size = (M, N, N)
    U : np.array
        size = (M, N, N)
        双対変数
    X : np.array
        size = (K, M, N, N)
    """

    def __init__(self, D, S, args):
        """
        Parameters
        ----------
        D : np.array
            size = (M, N, N)
            辞書の初期値
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
        self.dict_loop = args.dict_loop

        self.S = S
        self.D = D
        self.reset_parameters()

    def reset_parameters(self):
        self.G = self.D #制約条件D-G=0より
        self.U = np.zeros(shape=(self.M, self.N, self.N)) #双対変数Uの初期化
    
    def dict_update(self, X):
        """
        Parameters
        ----------
        X : np.array
            size = (K, M, N, N)
            係数マップ
        
        Returns
        -------
        D : np.array
            size = (M, N, N)
            最適化した辞書を返す
        """

        self.X = X

        self.update_D()
        self.update_G()
        self.update_U()

        return self.D

    def update_D(self):
        """
        辞書Dを更新する.
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
        G_hat = np.fft.fft2(self.G,  norm="ortho") #(M, N, N)
        U_hat = np.fft.fft2(self.U,  norm="ortho") #(M, N, N)

        #3.
        X_hat_conjugation = np.conjugate(X_hat) #(M, N, N)
        left_side = X_hat_conjugation*S_hat + self.Rho * (G_hat - U_hat) #(M, N, N)
        diag_inv = 1 / (1 + np.sum(X_hat * X_hat_conjugation, axis=0)/self.Rho) #(N, N)
        D_hat = left_side/self.Rho - X_hat_conjugation * (diag_inv * np.sum(X_hat * left_side, axis=0))/self.Rho


        #4.
        self.D = np.fft.ifft2(D_hat, norm="ortho").real
    
    def update_G(self):
        """
        Gを近接写像で更新する

        PP^{T}YについてはY[:, N-B:N, N-B:N]を0パディングすることで実現できる
        ノルムの正規化はYのノルムを見て1より大きかったら正規化する
        """

        Y = self.D + self.U

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
    
    def update_U(self):
        """
        Uを更新する
        """

        self.U = self.U + self.D - self.G