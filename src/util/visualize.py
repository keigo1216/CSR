import numpy as np
import matplotlib.pyplot as plt
from .rebuild import rebuild

def visualize(D, X, args, is_rebuild=True):
    """
    辞書と係数マップを可視化する

    Parameters
    ----------
    D : np.array
        size=(M, N, N)
        可視化する辞書
    X : np.array
        size=(K, M, N, N)
        可視化する係数マップ
    args : 各種パラメータ
    is_rebuild : boolen
        辞書と係数マップから再構築した画像を表示するかどうか
    -------
    """

    #辞書の可視化
    fig = plt.figure(figsize=(10*2, 10*(args.M//2)))
    for m in range(args.M):
        ax = fig.add_subplot(args.M//2, 2, m+1)
        ax.imshow(D[m, :args.B, :args.B], cmap="gray")
        ax.axis("off")
    plt.savefig("./out/Dic.png")

    #係数マップの可視化
    fig = plt.figure(figsize=(10*args.K, 10*(args.M+1)))
    for coef_map in range(args.M):
        for image in range(args.K):
            ax = fig.add_subplot(args.M+1, args.K, (coef_map)*(args.K)+(image+1))
            ax.imshow(X[image, coef_map], cmap="gray")
            ax.axis("off")
    
    if is_rebuild is True:
        for image in range(args.K):
            ax = fig.add_subplot(args.M+1, args.K, (args.M)*(args.K)+(image+1))
            ax.imshow(rebuild(D, X)[image], cmap="gray")
            ax.axis("off")

    plt.savefig("./out/Map.png")