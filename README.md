# Convolutional Sparse Reprezentation (CSR) Implementation
## Overview
CSR decompose the original image $\bm{s}_k \in \mathbb{R}^{N \times N}$ into M dictionaries $\bm{d}_m \in \mathbb{R}^{B \times B}$ and M coefficient maps $\bm{x}_{k,m} \in \mathbb{R}^{N \times N}$ on  $\bm{s}_k=\sum_m \bm{d}_m * \bm{x}_{k,m}$.

We can decompose images using ADMM in convex optimazation.