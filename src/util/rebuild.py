import numpy as np

def rebuild(D, X):
    """
    辞書と係数から画像を再構築する関数

    Parameters
    ----------
    D : np.array
        size=(M, N, N)
    X : np.array
        size=(K, M, N, N)
    
    Returns
    -------
    再構築した画像
    size = (K, N, N)
    """

    D_hat = np.fft.fft2(D, norm="ortho")
    X_hat = np.fft.fft2(X, norm="ortho")

    Image_hat = D_hat * X_hat
    Image = np.sum(np.fft.ifft2(Image_hat, norm="ortho"), axis=1).real

    return Image