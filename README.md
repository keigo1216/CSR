# Convolutional Sparse Reprezentation (CSR) Implementation
## Overview
We want to solve this convex optimazation problem. Decompose original images $s_k \in \mathbb{R}^{N \times N}$ to M dictionaries $d_m \in \mathbb{R}^{B \times B}$ and M coeffient map $x_{k,m} \in \mathbb{R}^{N \times N}$

$$ \rm{argmin}_{d_m, x_{k,m}} \frac{1}{2}\sum_k||\sum_m d_m * x_{k,m}-s_k||_2^2 + \lambda\sum_k\sum_m||x_{k,m}||_1$$

## 環境
- Python 3.10.6
- numpy 1.24.0
- matplotlib 3.6.2
- tqdm 4.64.1
