import numpy as np

def initialize_params(N, M, B):
    """
    辞書と係数マップを初期化する関数

    Parameters
    ----------
    N : int
        入力画像の縦（または横）の画素数
    M : int
        作成する辞書の枚数
    B : int
        辞書の縦（または横）の画素数

    Returns
    -------
    D : np.array
        size = (M, N, N)
        初期化された辞書
        パディングされた状態で出力
    
    X : np.array
        size = (M, N, N)
        係数マップ
    """

    D = np.random.normal(loc=0, scale=1, size=(M, B, B)) #平均0, 分散1の正規分布で初期化
    D = np.pad(D, ((0, 0), (0, N-B), (0, N-B))) #(:, N-B:N, N-B:N)を0パディング
    X = np.zeros(shape=(N, N))

    return D, X


if __name__ == "__main__":
    print(initialize_params(5, 2, 2))