from itertools import permutations
from math import factorial, dist, sqrt
from collections import Counter
from bisect import bisect_right, bisect_left, insort_left, insort_right

t = int(input())
ans = [0 for i in range(t)]

for i in range(t):
    n = int(input())
    p = list(map(int, input().split()))
    
    feasi_0 = True
    feasi_1 = False
    threshold_1 = 0
    test = 0
    
    for j in range(n):
        if feasi_0 and p[j] != j+1:
            feasi_0 = False
        if not feasi_1 and p[j] == j+1:
            if test == j * (j+1) // 2 :
                feasi_1 = True
        test += p[j]

    feasi_2 = True
    if p[0] == n and p[n-1] == 1:
        feasi_2 = False
        
    if feasi_0:
        ans[i] = 0
    elif feasi_1:
        ans[i] = 1
    elif feasi_2:
        ans[i] = 2
    else:
        ans[i] = 3
              
for i in range(t):
    print(ans[i])
    