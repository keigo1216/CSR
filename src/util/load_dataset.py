import numpy as np
from PIL import Image
import os

def load_dataset(path):
    """
    指定したディレクトリにある画像を取得する

    parameters
    ----------
    path : str
           画像が入っているディレクトリのpath
           
    returns
    -------
    K : int
        取得した画像の枚数
    S : np.array
        size = (K, N, N)
    """
    files = os.listdir(path=path)
    S = []
    for file_name in files:
        if not os.path.isfile(os.path.join(path, file_name)): #ファイル名がディレクトリの場合はスルーする
            continue

        S.append(np.array(Image.open(os.path.join(path, file_name)))/255) #画像を取得してSにくっつける

    return len(S), np.array(S)

if __name__ == "__main__":
    print(load_dataset("../../dataset").shape)