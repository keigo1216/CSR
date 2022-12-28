import numpy as np 
import matplotlib.pyplot as plt
import argparse

from util.initialize_params import initialize_params
from util.load_dataset import load_dataset
from util.visualize import visualize

from optim.CSR_optim import CSR_optim

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="畳み込みスパース表現を実装したプログラムです. 各変数の値は黒木様の資料p3を参考にしてください. ")
    parser.add_argument("--N",         type=int, default=128,            help="画像の縦（または横）の画素数. 黒木様の資料は全画素数でやっていますので注意. ")
    parser.add_argument("--M",         type=int, default=8,              help="辞書の枚数")
    parser.add_argument("--B",         type=int, default=16,             help="辞書の縦（または横）の画素数. 黒木様の資料は全画素数でやっていますので注意. ")
    parser.add_argument("--K",         type=int, default=1,              help="学習枚数")
    parser.add_argument("--Rho",       type=int, default=1,              help="ADMMの係数ロー")
    parser.add_argument("--Lam",       type=int, default=0.001,          help="最適化する時のハイパーパラメータ")
    parser.add_argument("--data_dir",  type=str, default= "../dataset",  help="データセットのディレクトリ")
    parser.add_argument("--coef_loop", type=int, default=10,              help="係数マップの最適化回数")
    parser.add_argument("--dict_loop", type=int, default=10,              help="辞書の最適化回数")
    parser.add_argument("--hole_loop", type=int, default=1000,           help="全体の最適化回数")
    parser.add_argument("--is_consensus", type=bool, default=True,       help="最適化する際にコンセンサス方式を使うかどうか")
    args = parser.parse_args()

    K, S = load_dataset(args.data_dir) #データセットの取得.
    args.K = K
    D, X = initialize_params(args.N, args.M, args.B, args.K) #辞書と係数を初期化

    csr_optim = CSR_optim(D, X, S, args)

    for i in tqdm(range(args.hole_loop)):
        D, X = csr_optim.optimization()

    visualize(D, X, args, True) #画像と辞書を表示させる