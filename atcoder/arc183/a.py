import itertools
import math

def find_kth_good_sequence(n, k):
    # 辞書順で S 個の全ての良い整数列を生成
    # S = total number of good sequences
    S = math.factorial(n * k) // (math.factorial(k) ** n)
    
    # floor((S+1)/2) 番目のシーケンスを求める
    target_index = (S + 1) // 2 
    
    # 初期状態のシーケンス
    sequence = [i for i in range(1, n+1) for _ in range(k)]
    
    # 全ての良いシーケンスを生成し、辞書順でtarget_index番目を見つける
    for idx, perm in enumerate(itertools.permutations(sequence)):
        if idx == target_index:
            return perm

# 例として N=3, K=2の場合
N = int(input("Nを入力: "))
K = int(input("Kを入力: "))
result = find_kth_good_sequence(N, K)
print("結果の良い整数列:", result)
