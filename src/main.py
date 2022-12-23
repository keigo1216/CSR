import numpy as np 
import matplotlib.pyplot as plt
import argparse

from util.initialize_params import initialize_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="畳み込みスパース表現を実装したプログラムです. 各変数の値は黒木様の資料p3を参考にしてください. ")
    parser.add_argument("--N",         type=int, default=128*128, help="画像の縦（または横）の画素数. 黒木様の資料は全画素数でやっていますので注意. ")
    parser.add_argument("--M",         type=int, default=10,      help="辞書の枚数")
    parser.add_argument("--B",         type=int, default=12*12,   help="辞書の縦（または横）の画素数. 黒木様の資料は全画素数でやっていますので注意. ")
    parser.add_argument("--coef_loop", type=int, default=2,       help="係数マップの最適化回数")
    parser.add_argument("--dict_loop", type=int, default=2,       help="辞書の最適化回数")
    args = parser.parse_args()

    D, X = initialize_params(args.N, args.M, args.B) #辞書と係数を初期化