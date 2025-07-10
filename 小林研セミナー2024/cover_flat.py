## ライブラリのimport
# pip install mip 
import sys
# sys.path.append("/home/yudai/.local/lib/python3.10/site-packages")
import mip
# from mip import Model, maximize, minimize, xsum
import numpy as np
from numpy import pi
from numpy.random import rand, randn
from numpy.linalg import solve, det
from scipy.linalg import sqrtm
import math
import statistics

## 定数の設定
p = 2 # 行列の次元
N = 500 # データ数
delta = 10.0
epsilon = 1.0
h = 4.0 #U(-h,h)
## Wasserstein距離を計算する関数
def wasserstein(x, y):
    z = sqrtm(sqrtm(x) @ y @ sqrtm(x))
    res = np.trace(x) + np.trace(y) - 2 * np.trace(z)
    return np.sqrt(res)

## dist_flat
def flat_dist(x, y):
    res = np.sqrt(np.trace((x-y).T @ (x-y)))
    return res

# 各シードでのカバリングナンバーを格納
cn_flat = []
cn_bw = []

for seed in range(10):    
    ## 乱数の生成と距離グラフの生成
    ### wasserstein
    P = []
    nbh_w = [[] for _ in range(N)]

    i = 0
    np.random.seed(seed)
    while i < N:
        a = np.array([[rand(1) * 2.0 * h - h, rand(1) * 2.0 * h - h],[rand(1) * 2.0 * h - h, rand(1) * 2.0 * h * h]])
        a = a.reshape((2,2)) 
        v = a @ a.T
        # B(I,delta)内になければskip
        if det(v) < 0.0001:
            continue 
        if wasserstein(np.eye(p),v) > delta :
            continue
        
        for j in range(0,i):
            w_dist = wasserstein(P[j],v)
            if w_dist < epsilon:
                nbh_w[i].append(j)
                nbh_w[j].append(i)
                
        # print(i)    
        nbh_w[i].append(i)
                
        i += 1
        P.append(v)
        
    ### flat
    P = []
    nbh_rw = [[] for _ in range(N)]


    i = 0
    np.random.seed(123)
    while i < N:
        a = np.array([[rand(1) * 2.0 * h - h, rand(1) * 2.0 * h - h],[rand(1) * 2.0 * h - h, rand(1) * 2.0 * h * h]])
        a = a.reshape((2,2)) 
        v = a @ a.T
        # B(I,delta)内になければskip
        if flat_dist(np.eye(p),v) > delta:
            continue
        
        for j in range(0,i):
            rw_dist = flat_dist(P[j],v)
            if rw_dist < epsilon:
                nbh_rw[i].append(j)
                nbh_rw[j].append(i)
                
        # print(i)    
        nbh_rw[i].append(i)
                
        i += 1
        P.append(v)
        
    ## Covering number
    ### Wasserstein
    from mip import Model, maximize, minimize, xsum
    m = Model()  # 数理モデル
    x = []
    # 変数
    for i in range(N):
        s = "x"+str(i)
        x.append(m.add_var(s, lb=0, var_type="I"))

    # 目的関数
    z = x[0]
    for i in range(1,N):
        z += x[i]

    m.objective = minimize(z)
    # m.objective = maximize(100 * x + 100 * y)
    # 制約条件
    for i in range(N):
        b = 0
        for j in nbh_w[i]:
            b += x[j]
        m += b >= 1
        
    m.optimize()  # ソルバーの実行
    print(m.objective.x)
    cn_bw.append(m.objective.x)

    # print("obj", m.objective.x, "x", x.x, "y", y.x)
    ### flat
    m = Model()  # 数理モデル
    x = []
    # 変数
    for i in range(N):
        s = "x"+str(i)
        x.append(m.add_var(s, lb=0, var_type="I"))

    # 目的関数
    z = x[0]
    for i in range(1,N):
        z += x[i]

    m.objective = minimize(z)
    # m.objective = maximize(100 * x + 100 * y)
    # 制約条件
    for i in range(N):
        b = 1
        for j in nbh_rw[i]:
            b += x[j]
        m += b >= 2
        
    m.optimize()  # ソルバーの実行
    print(m.objective.x)
    cn_flat.append(m.objective.x)

print("bw: mean:" + statistics.mean(cn_bw) + " std: " + statistics.pstdev(cn_bw))    
print("flat: mean:" + statistics.mean(cn_flat) + " std: " + statistics.pstdev(cn_flat))    